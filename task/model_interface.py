# model_interface.py
# User-customizable subclass where you implement model-specific logic and steps

from typing import Iterator, Optional, Dict
import torch
from bionemo.llm.api import MegatronModelType, MegatronLossType
from src.interface.model_interface_base import ModelInterfaceBase
from src.model.foldrep_model import FoldRepModelConfig, FoldRepModel
from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from typing import Iterator, Optional, Dict, Any
# from task.loss import compute_custom_loss
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from megatron.core.optimizer import OptimizerConfig
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
# from src.data.protein import Protein
from torch.cuda import empty_cache


class BionemoLightningModule(
    ModelInterfaceBase[MegatronModelType, MegatronLossType]
):
    """User implementation: override only these methods to define your model."""
    def __init__(self, config) -> None:
        """Initialize module from a unified config bundle only."""
        super().__init__(
            model_transform=None,
            configure_init_model_parallel=False,
        )
        # Store original bundle
        self.config_bundle = config
        # Copy required fields onto self to avoid using read-only hparams property
        self.enc_layers = config.model.enc_layers
        self.dec_layers = config.model.dec_layers
        self.hidden_dim = config.model.hidden_dim
        self.prefix_len = config.model.prefix_len
        self.warmup_steps = config.training.warmup_steps
        self.lr = config.training.lr
        self.scheduler_num_steps = (
            config.training.scheduler_num_steps if config.training.scheduler_num_steps is not None else config.training.num_steps
        )
        self.custom_checkpoint_path = config.experiment.custom_checkpoint_path
        self.infer_feats = config.experiment.infer_feats
        self.d_model = config.model.hidden_size
        self.n_heads = config.model.num_attention_heads
        self.nn_neighbors = config.model.nn_neighbors
        self.input_modality = config.model.input_modality
        self.output_modality = config.model.output_modality
        self.use_dino = getattr(config.model, 'use_dino', 0)
        optimizer = self.set_optimizer()
        self.optim = optimizer
        self.optim.connect(self)
        self.config = self.set_config()
        # DINO / EMA params
        self.dino_teacher_momentum: float = 0.996
        self.dino_teacher_temp: float = 0.04
        self.dino_student_temp: float = 0.1
        self.dino_center_momentum: float = 0.9
        self.dino_center: Optional[torch.Tensor] = None
        self.dino_weight: float = 1.0
        self.teacher_module: Optional[torch.nn.Module] = None
        
    def set_config(self):
        self.config = FoldRepModelConfig(
            enc_layers=self.enc_layers,
            dec_layers=self.dec_layers,
            hidden_dim=self.hidden_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            nn_neighbors=self.nn_neighbors,
            input_modality=self.input_modality,
            output_modality=self.output_modality,
            prefix_len=self.prefix_len,
            dropout=0.0,
            max_seq_length=1024,
            num_attention_heads=self.n_heads,
            num_layers=max(self.enc_layers, self.dec_layers),
            hidden_size=self.d_model,
            use_dino=self.use_dino,
        )
        return self.config

    def configure_model(self) -> None:
        """Instantiate the FoldRepModel and assign to self.module"""
        self.module = FoldRepModel(self.config)
        if self.custom_checkpoint_path != "":
            # Load from Megatron checkpoint format
            self.load_from_megatron_ckpt(self.custom_checkpoint_path)
        # Initialize EMA teacher after student is created
        if self.use_dino==1:
            self._init_teacher()

    def _init_teacher(self) -> None:
        if self.teacher_module is None:
            # Create an EMA teacher as a frozen copy
            import copy
            self.teacher_module = copy.deepcopy(self.module).eval()
            for p in self.teacher_module.parameters():
                p.requires_grad = False
    
    
    def loss_reduction(self, *args, **kwargs):
        # 返回 CustomLossWithReduction 类的实例
        from src.model.foldrep_model import CustomLossWithReduction
        return CustomLossWithReduction(self.module.module.module, **kwargs)

    def data_step(self, dataloader_iter: Iterator) -> Dict:
        """Move batch to GPU and select the correct parallel slice."""
        batch = next(dataloader_iter)
        if isinstance(batch, tuple) and len(batch) == 3:
            _batch = batch[0]
        else:
            _batch = batch
        def to_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            elif isinstance(x, (list, tuple)):
                return [to_cuda(i) for i in x]
            elif isinstance(x, dict):
                return {k: to_cuda(v) for k, v in x.items()}
            else:
                return x
        _batch = {k: to_cuda(v) for k, v in _batch.items()}
        _batch['use_dino'] = self.use_dino
        return get_batch_on_this_context_parallel_rank(_batch)

    def forward_step(self, batch: Dict, infer_feats=False) -> Dict:
        """Core forward: build attention mask, compute features, and run the model."""
        data_id = batch['data_id']
        attn_mask = (
            (data_id[:, :, None] == data_id[:, None, :])
            & (data_id[:, :, None] >= 0)
            & (data_id[:, None, :] >= 0)
        )
        dummy_node = (data_id == -1)[..., None]
        attn_mask = (attn_mask | dummy_node) & ~dummy_node.transpose(1, 2)
        

        
        pred_out = self.module(
            batch['position'],
            batch['seq_ids'],
            batch['blocks'],
            attn_mask,
            infer_feats=infer_feats
        )
        
        # forward_out = {'pred_out': pred_out, 'batch': batch}
        return pred_out

    @torch.no_grad()
    def _ema_update_teacher(self) -> None:
        if self.teacher_module is None:
            return
        momentum = self.dino_teacher_momentum
        for p_student, p_teacher in zip(self.module.parameters(), self.teacher_module.parameters()):
            p_teacher.data.mul_(momentum).add_(p_student.data, alpha=1.0 - momentum)

    def _sample_select(self, batch: Dict, keep_prob: float = 0.8) -> torch.Tensor:
        # Randomly select a sub-sequence (possibly disjoint) via boolean mask
        device = batch['seq_ids'].device
        B, L = batch['seq_ids'].shape
        # We build one mask per global sequence since data is concatenated; build token-wise Bernoulli with min length constraints
        # Keep prefix tokens always
        prefix_len = int(batch.get('prefix_len', getattr(self, 'prefix_len', 0)))
        rand_keep = torch.rand(B, L, device=device) < keep_prob
        # Always keep padding/dummy tokens off
        valid = batch['data_id'] > 0
        select = rand_keep & valid
        # Ensure prefix kept to help alignment
        if prefix_len > 0:
            prefix_mask = torch.zeros_like(select)
            prefix_mask[:, :prefix_len] = True
            select = select | prefix_mask
        return select

    def _apply_select(self, batch: Dict, select: torch.Tensor) -> Dict:
        # Mask out tokens not selected by setting data_id to -1 and zeroing tensors
        view = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        mask_out = ~select
        if 'data_id' in view:
            view['data_id'] = view['data_id'].clone()
            view['data_id'][mask_out] = -1
        if 'seq_ids' in view:
            view['seq_ids'] = view['seq_ids'].clone()
            view['seq_ids'][mask_out] = 21 # mask token
        if 'blocks' in view:
            view['blocks'] = view['blocks'].clone()
            view['blocks'][mask_out] = 0
        if 'position' in view:
            view['position'] = view['position']
        return view

    def _compute_feats(self, module: torch.nn.Module, batch_view: Dict) -> torch.Tensor:
        data_id = batch_view['data_id']
        attn_mask = (
            (data_id[:, :, None] == data_id[:, None, :])
            & (data_id[:, :, None] >= 0)
            & (data_id[:, None, :] >= 0)
        )
        dummy_node = (data_id == -1)[..., None]
        attn_mask = (attn_mask | dummy_node) & ~dummy_node.transpose(1, 2)
        feats = module(
            batch_view['position'],
            batch_view['seq_ids'],
            batch_view['blocks'],
            attn_mask,
            infer_feats=True,
        )
        return feats

    def _compute_dino_loss(self, feats_s: torch.Tensor, feats_t: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
        # feats: [L, D] or [B,L,D]? model returns [L,D]; in our pipeline it is [L,D]
        # Align on overlapping tokens
        if overlap.sum() == 0:
            return feats_s.new_zeros(())
        s = feats_s[overlap]
        t = feats_t[overlap]
        # Temperatures and centering
        if self.dino_center is None:
            self.dino_center = t.mean(dim=0, keepdim=True).detach()
        t_cent = (t - self.dino_center)
        logits_s = s / self.dino_student_temp
        with torch.no_grad():
            logits_t = t_cent / self.dino_teacher_temp
            probs_t = torch.softmax(logits_t, dim=-1)
        log_probs_s = torch.log_softmax(logits_s, dim=-1)
        loss = -(probs_t * log_probs_s).sum(dim=-1).mean()
        # Update center
        with torch.no_grad():
            batch_center = t.mean(dim=0, keepdim=True)
            self.dino_center = self.dino_center * self.dino_center_momentum + batch_center * (1.0 - self.dino_center_momentum)
        return loss * self.dino_weight

    def _compute_dino_global_loss(self, feats_s: torch.Tensor, feats_t: torch.Tensor) -> torch.Tensor:
        # feats_s, feats_t: [B, D] global representations already normalized upstream
        if feats_s.ndim != 2 or feats_t.ndim != 2:
            feats_s = feats_s.reshape(feats_s.shape[0], -1)
            feats_t = feats_t.reshape(feats_t.shape[0], -1)

        assert feats_s.shape == feats_t.shape, "Student/Teacher global features must match"
        # Temperatures and centering
        if self.dino_center is None:
            self.dino_center = feats_t.mean(dim=0, keepdim=True).detach()
        logits_s = feats_s / self.dino_student_temp
        with torch.no_grad():
            logits_t = (feats_t - self.dino_center) / self.dino_teacher_temp
            probs_t = torch.softmax(logits_t, dim=-1)
        log_probs_s = torch.log_softmax(logits_s, dim=-1)
        loss = -(probs_t * log_probs_s).sum(dim=-1).mean()
        # Update center using teacher batch stats
        with torch.no_grad():
            batch_center = feats_t.mean(dim=0, keepdim=True)
            self.dino_center = self.dino_center * self.dino_center_momentum + batch_center * (1.0 - self.dino_center_momentum)
        return loss * self.dino_weight

    def _compute_contrastive_loss(self, feats_s1: torch.Tensor, feats_s2: torch.Tensor) -> torch.Tensor:
        # feats_s1, feats_s2: [B, D] global representations already normalized upstream
        batch_size = feats_s1.size(0)
        
        # Positive pairs: feats_s1 and feats_s2 (same sample, different views)
        pos_sim = torch.nn.functional.cosine_similarity(feats_s1, feats_s2, dim=-1)
        
        # Generate negative samples by shuffling feats_s2
        # Create random permutation indices
        neg_indices = torch.randperm(batch_size, device=feats_s1.device)
        feats_s2_neg = feats_s2[neg_indices]  # Shuffled feats_s2 for negative pairs
        
        # Negative pairs: feats_s1 and shuffled feats_s2 (different samples)
        neg_sim = torch.nn.functional.cosine_similarity(feats_s1, feats_s2_neg, dim=-1)
        
        # Contrastive loss: maximize positive similarity, minimize negative similarity
        # Using InfoNCE-style loss: -log(exp(pos_sim/τ) / (exp(pos_sim/τ) + exp(neg_sim/τ)))
        temperature = 0.07  # Temperature parameter for contrastive learning
        pos_logits = pos_sim / temperature
        neg_logits = neg_sim / temperature
        
        # Compute log probabilities
        log_prob_pos = pos_logits - torch.logsumexp(torch.stack([pos_logits, neg_logits], dim=-1), dim=-1)
        
        # Contrastive loss (negative log likelihood)
        loss = -log_prob_pos.mean()*10
        
        return loss

    def _random_mlm_mask(self, batch: Dict, mask_ratio: float = 0.15) -> Dict:
        """
        对蛋白质序列进行随机mask指定比例的采样
        
        Args:
            batch: 包含蛋白质序列数据的批次字典
            mask_ratio: mask的比例，默认0.15 (15%)
        
        Returns:
            修改后的batch，其中部分氨基酸被mask
        """
        device = batch['seq_ids'].device
        B, L = batch['seq_ids'].shape
        batch['seq_ids_ori'] = batch['seq_ids'].clone()
        batch['blocks_ori'] = batch['blocks'].clone()
        
        # 创建mask后的batch副本
        masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 初始化masked_position tensor，用于记录哪些位置被mask了
        masked_batch['masked_position'] = torch.zeros(B, L, dtype=torch.bool, device=device)
        
        # 获取有效token的mask (排除padding和特殊token)
        valid_mask = batch['data_id'] > 0
        
        # 为每个序列生成随机mask
        for b in range(B):
            # 获取当前序列的有效位置
            valid_positions = torch.where(valid_mask[b])[0]
            if len(valid_positions) == 0:
                continue
                
            # 计算需要mask的token数量
            num_valid = len(valid_positions)
            num_to_mask = int(num_valid * mask_ratio)
            
            if num_to_mask == 0:
                continue
                
            # 随机选择要mask的位置
            mask_indices = torch.randperm(num_valid, device=device)[:num_to_mask]
            mask_positions = valid_positions[mask_indices]
            
            # 应用mask
            # 将选中的位置设置为mask token (通常为21)
            masked_batch['seq_ids'][b, mask_positions] = 21
            
            # 记录被mask的位置，用于计算loss
            masked_batch['masked_position'][b, mask_positions] = True
            
            # 可选：将对应的结构信息置为0
            if 'blocks' in masked_batch:
                masked_batch['blocks'][b, mask_positions] = 0
                # masked_batch['blocks'][b, mask_positions].mean(dim=0)
        
        return masked_batch

    def _advanced_mlm_mask(self, batch: Dict, mask_ratio: float = 0.15, 
                          replace_prob: float = 0.8, 
                          random_prob: float = 0.1, 
                          keep_prob: float = 0.1) -> Dict:
        """
        高级MLM mask功能，支持不同的mask策略
        
        Args:
            batch: 包含蛋白质序列数据的批次字典
            mask_ratio: mask的比例，默认0.15 (15%)
            replace_prob: 被mask的token中，用[MASK]替换的比例，默认0.8
            random_prob: 被mask的token中，用随机氨基酸替换的比例，默认0.1  
            keep_prob: 被mask的token中，保持原token的比例，默认0.1
        
        Returns:
            修改后的batch，其中部分氨基酸被mask
        """
        device = batch['seq_ids'].device
        B, L = batch['seq_ids'].shape
        batch['seq_ids_ori'] = batch['seq_ids'].clone()
        batch['blocks_ori'] = batch['blocks'].clone()
        prefix_len = (batch['seq_ids'][0]==34).sum().item()
        # 验证概率和
        assert abs(replace_prob + random_prob + keep_prob - 1.0) < 1e-6, \
            f"概率和必须为1，当前为: {replace_prob + random_prob + keep_prob}"
        
        # 创建mask后的batch副本
        masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 初始化masked_position tensor，用于记录哪些位置被mask了
        masked_batch['masked_position'] = torch.zeros(B, L, dtype=torch.bool, device=device)
        
        # 获取有效token的mask (排除padding和特殊token)
        valid_mask = batch['data_id'] > 0
        valid_mask[:,:prefix_len]=False
        
        # 氨基酸词汇表 (0-20，其中21是[MASK] token)
        amino_acids = torch.arange(0, 21, device=device)
        
        # 为每个序列生成随机mask
        for b in range(B):
            # 获取当前序列的有效位置
            valid_positions = torch.where(valid_mask[b])[0]
            if len(valid_positions) == 0:
                continue
                
            # 计算需要mask的token数量
            num_valid = len(valid_positions)
            num_to_mask = int(num_valid * mask_ratio)
            
            if num_to_mask == 0:
                continue
                
            # 随机选择要mask的位置
            mask_indices = torch.randperm(num_valid, device=device)[:num_to_mask]
            mask_positions = valid_positions[mask_indices]
            
            # 获取原始token
            original_tokens = batch['seq_ids'][b, mask_positions]
            
            # 生成随机数来决定每个token的处理方式
            rand_vals = torch.rand(num_to_mask, device=device)
            
            # 计算累积概率
            cum_replace_prob = replace_prob
            cum_random_prob = replace_prob + random_prob
            
            # 应用不同的mask策略
            replace_mask = rand_vals < cum_replace_prob
            random_mask = (rand_vals >= cum_replace_prob) & (rand_vals < cum_random_prob)
            keep_mask = rand_vals >= cum_random_prob
            
            # 记录所有被选中的位置（无论采用哪种策略）
            masked_batch['masked_position'][b, mask_positions] = True
            
            # 1. 用[MASK] token替换
            if replace_mask.any():
                masked_batch['seq_ids'][b, mask_positions[replace_mask]] = 21
            
            # 2. 用随机氨基酸替换
            if random_mask.any():
                random_tokens = torch.randint(0, 21, (random_mask.sum(),), device=device)
                masked_batch['seq_ids'][b, mask_positions[random_mask]] = random_tokens
            
            # 3. 保持原token (keep_mask已经处理，不需要额外操作)
            
            # 可选：将对应的结构信息也置0
            if 'blocks' in masked_batch:
                masked_batch['blocks'][b, mask_positions] = 0
                
                # masked_batch['blocks'][b, mask_positions].mean(dim=0)
        
        
        return masked_batch

    def training_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        """Training step: set prefix length and run forward_step."""
        batch['prefix_len'] = self.prefix_len
        
        # R = random_rotation_matrix()[0]
        # t = torch.rand(1,1,1,3).cuda().to(R.dtype)+3
        # batch['coords'] = torch.einsum('blki, ij->blkj', batch['coords'], R.cuda())+t
        # batch['blocks'] = torch.einsum('blki, ij->blkj', batch['blocks'], R.cuda())+t
        # outputs2 = self.forward_step(batch)
        # loss2, results2 = compute_custom_loss(outputs2, batch)
        # DINO: build two random views and compute teacher-student alignment
        
        if self.use_dino==1:

            select1 = self._sample_select(batch, keep_prob=0.8)
            select2 = self._sample_select(batch, keep_prob=0.8)
            view1 = self._apply_select(batch, select1)
            view2 = self._apply_select(batch, select2)
            # Student features
            feats_s1 = self.forward_step(view1, infer_feats=1)
            feats_s2 = self.forward_step(view2, infer_feats=1)
            # Teacher features (no grad)
            with torch.no_grad():
                self.teacher_module.eval()
                feats_t1 = self._compute_feats(self.teacher_module, view1)
                feats_t2 = self._compute_feats(self.teacher_module, view2)
            
            # Overlap masks
            # valid = (batch['data_id'] > 0)
            # overlap12 = (select1 & select2 & valid)
            # # Two-way loss (s1 vs t2) and (s2 vs t1)
            # dino_loss = self._compute_dino_loss(feats_s1, feats_t2, overlap12)
            # dino_loss = dino_loss + self._compute_dino_loss(feats_s2,  feats_t1, overlap12)
            
            # Two-way global loss (s1 vs t2, s2 vs t1) 
            dino_loss = self._compute_dino_global_loss(feats_s1, feats_t2)
            dino_loss = dino_loss + self._compute_dino_global_loss(feats_s2, feats_t1)
            batch['dino_loss'] = dino_loss
        elif self.use_dino==2: # contrastive loss
            select1 = self._sample_select(batch, keep_prob=1.0)
            select2 = self._sample_select(batch, keep_prob=0.8)
            view1 = self._apply_select(batch, select1)
            view2 = self._apply_select(batch, select2)
            # Student features
            feats_s1 = self.forward_step(view1, infer_feats=2)
            feats_s2 = self.forward_step(view2, infer_feats=2)
            contrastive_loss = self._compute_contrastive_loss(feats_s1, feats_s2)
            batch['contrastive_loss'] = contrastive_loss
        elif self.use_dino==3: # mlm baseline
            batch = self._advanced_mlm_mask(batch, mask_ratio=0.15)
        elif self.use_dino==4: # masked struct + compression baseline
            batch = self._advanced_mlm_mask(batch, mask_ratio=0.15)
            


        outputs = self.forward_step(batch)
        
        # 计算损失用于日志记录
        if self.is_on_logging_device():
            loss, results = self.module.module.module.compute_custom_loss(outputs, batch)
            for key, val in results.items():
                self.log("train_"+key, val.detach().item(), on_step=True, on_epoch=False, prog_bar=True)
        
        # # idx = 2
        # for idx in range(32):
        #     true_X = batch['coords'][:,:,:5]
        #     mask0 = true_X.sum(dim=(-2,-1))!=0
        #     mask = (batch['data_id']>0)&mask0
        #     pred_X = outputs['predX']
        #     X = pred_X[idx][mask[idx]][None][:,:,[0,1,2,4]]
        #     C = torch.ones_like(X)[:,:,0,0].long()
        #     protein_pred = Protein.from_XCS(X, C, C)
            
        #     X = true_X[idx][mask[idx]][None][:,:,[0,1,2,4]]
        #     protein_true = Protein.from_XCS(X, C, C)
            
        #     protein_pred.to(f'/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/sample{idx}_pred.pdb')
        #     protein_true.to(f'/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/sample{idx}_true.pdb')
        # EMA teacher update
        if self.use_dino==1:
            self._ema_update_teacher()
        
        return outputs

    def validation_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        # torch.cuda.empty_cache()
        """Validation step: set prefix length, eval mode, and run forward_step without gradient."""

        batch['prefix_len'] = self.prefix_len
        with torch.no_grad():
            self.module.eval()
            if self.use_dino==1:
                select1 = self._sample_select(batch)
                select2 = self._sample_select(batch)
                view1 = self._apply_select(batch, select1)
                view2 = self._apply_select(batch, select2)
                # Student features
                feats_s1 = self.forward_step(view1, infer_feats=1)
                feats_s2 = self.forward_step(view2, infer_feats=1)
                # Teacher features (no grad)
                with torch.no_grad():
                    self.teacher_module.eval()
                    feats_t1 = self._compute_feats(self.teacher_module, view1)
                    feats_t2 = self._compute_feats(self.teacher_module, view2)
                
                # Overlap masks
                # valid = (batch['data_id'] > 0)
                # overlap12 = (select1 & select2 & valid)
                # # Two-way loss (s1 vs t2) and (s2 vs t1)
                # dino_loss = self._compute_dino_loss(feats_s1, feats_t2, overlap12)
                # dino_loss = dino_loss + self._compute_dino_loss(feats_s2,  feats_t1, overlap12)
                
                # Two-way global loss (s1 vs t2, s2 vs t1) 
                dino_loss = self._compute_dino_global_loss(feats_s1, feats_t2)
                dino_loss = dino_loss + self._compute_dino_global_loss(feats_s2, feats_t1)
                batch['dino_loss'] = dino_loss
            elif self.use_dino==2: # contrastive loss
                select1 = self._sample_select(batch, keep_prob=1.0)
                select2 = self._sample_select(batch, keep_prob=0.8)
                view1 = self._apply_select(batch, select1)
                view2 = self._apply_select(batch, select2)
                # Student features
                feats_s1 = self.forward_step(view1, infer_feats=2)
                feats_s2 = self.forward_step(view2, infer_feats=2)
                contrastive_loss = self._compute_contrastive_loss(feats_s1, feats_s2)
                batch['contrastive_loss'] = contrastive_loss
            elif self.use_dino==3: # mlm baseline
                batch = self._advanced_mlm_mask(batch, mask_ratio=0.15)

            
            outputs = self.forward_step(batch)
            if self.is_on_logging_device():
                loss, results = self.module.module.module.compute_custom_loss(outputs, batch)
                for key, val in results.items():
                    if key != 'loss':
                        self.log("val_"+key, val, on_step=True, on_epoch=False, prog_bar=True)
            # self.log_dict({f'val_loss': results['loss']})
            return outputs
    

    def predict_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Optional[Dict]:
        """Predict step alias to forward_step for inference."""
        if not batch:
            return None
        return self.forward_step(batch)

    def set_optimizer(self):
        optimizer = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=self.lr,
                optimizer="adam",
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
                clip_grad=1.0,
                adam_eps=1e-8
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=self.warmup_steps,
                max_steps=self.scheduler_num_steps,
                max_lr=self.lr,
                min_lr=0.0,
                anneal_percentage=0.01,
            ),
        )
        return optimizer

    def save_to_torch_ckpt(self, ckpt_dir: str, out_path: str) -> None:
        """Save the model to a torch checkpoint."""
        '''
        self.save_to_torch_ckpt('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1/checkpoints/epoch=0-step=94999-consumed_samples=760000.0/weights', '/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1/checkpoints/epoch=0-step=94999-consumed_samples=760000.0/model.pt')
        '''
        
        # 1. 生成 sharded_state_dict
        sharded_sd = self.module.sharded_state_dict()  # <— 一定要在 parallel 初始化后调用
        ckpt = self.trainer.strategy.checkpoint_io.load_checkpoint(
                str(ckpt_dir),
                sharded_state_dict=sharded_sd,            # <<< 关键：这里不能省略
            )  # 底层会调用 dist_checkpointing.load(sharded_state_dict=…, …)
        torch.save(ckpt, out_path)
        print(f"✅ 转换成功：{out_path}")
    
    def load_from_torch_ckpt(self, ckpt_path: str) -> None:
        """Load the model from a torch checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt, strict=False)
        print(f"✅ 模型加载成功：{ckpt_path}")
    
    def load_from_megatron_ckpt(self, ckpt_dir: str) -> None:
        """Load the model from a Megatron checkpoint directory."""
        # Initialize process group if needed
        self.init_process_group_if_needed(backend="gloo")
        
        # Load the Megatron checkpoint
        from megatron.core.dist_checkpointing import load_plain_tensors
        full_state_dict = load_plain_tensors(ckpt_dir)
        
        # Extract only the model state dict from the Lightning checkpoint
        model_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('module.') and not key.startswith('optimizer.'):
                model_state_dict[key] = value
        
        # Load the model state dict into the model
        self.load_state_dict(model_state_dict, strict=True)
        print(f"✅ Megatron checkpoint loaded successfully: {ckpt_dir}")
    
    # def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
    #     """Hook to clear CUDA cache before each optimizer step."""
    #     # replace gradient nan/inf with 0
    #     for group in optimizer.param_groups:
    #         for param in group['params']:
    #             if param.grad is not None:
    #                 grad = param.grad
    #                 # 将 NaN 和 Inf 替换为 0
    #                 grad = torch.where(torch.isnan(grad) | torch.isinf(grad), torch.zeros_like(grad), grad)
    #                 param.grad = grad
    
    def find_available_port(self, start_port=12355, max_attempts=100):
        """
        查找可用端口，从start_port开始尝试
        """
        import socket
        import random
        
        for _ in range(max_attempts):
            try:
                # 随机选择一个端口范围，避免冲突
                port = start_port + random.randint(0, 1000)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        # 如果都不可用，使用系统分配的端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]

    def init_process_group_if_needed(self, backend="gloo"):
        import torch.distributed as dist
        import os
        
        # 动态查找可用端口
        available_port = self.find_available_port()
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(available_port)
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        if not dist.is_initialized():
            # 单进程用 gloo 足够
            dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0)
            print(f"✅ 分布式进程组初始化成功，使用端口: {available_port}")

    def extract_plain_state(self, ckpt_dir: str, output_path: str):
        from megatron.core.dist_checkpointing import load_plain_tensors
        # ⚠️ 一定要先初始化分布式 group
        self.init_process_group_if_needed(backend="gloo")
        # 加载 shard checkpoint 自动合并
        state_dict = load_plain_tensors(ckpt_dir)
        torch.save(state_dict, output_path)
        print(f"✔️ Saved merged checkpoint at: {output_path}")
        return state_dict

import os
import lmdb
import torch
import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()  # 启用对 numpy 的支持

def ensure_lmdb_dir(lmdb_path):
    """
    确保 LMDB 存储目录存在
    """
    dir_path = os.path.dirname(os.path.abspath(lmdb_path))
    os.makedirs(dir_path, exist_ok=True)

def save_vectors_to_lmdb(data_dict, lmdb_path):
    """
    将多个 PyTorch 向量写入 LMDB（使用 msgpack）
    :param data_dict: dict[str -> torch.Tensor]
    :param lmdb_path: 存储路径（如 './data/my_vectors.lmdb'）
    """
    ensure_lmdb_dir(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=1 << 40)  # 最大约 1 TB
    with env.begin(write=True) as txn:
        for name, tensor in data_dict.items():
            # 将 tensor 转为 numpy，使用 msgpack 序列化
            array = tensor.cpu().numpy()
            serialized = msgpack.packb(array, default=m.encode, use_bin_type=True)
            txn.put(name.encode('utf-8'), serialized)
    env.close()

def load_vector_from_lmdb(name, lmdb_path):
    """
    根据 name 从 LMDB 中读取向量（使用 msgpack）
    :param name: str
    :param lmdb_path: str
    :return: torch.Tensor or None
    """
    if not os.path.exists(lmdb_path):
        print(f"LMDB path '{lmdb_path}' 不存在。")
        return None

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        value = txn.get(name.encode('utf-8'))
        if value is None:
            return None
        array = msgpack.unpackb(value, raw=False, object_hook=m.decode)
        return torch.from_numpy(array)


def random_rotation_matrix(batch_size: int = 1, device=None, dtype=torch.float32):
    """
    生成 batch_size 个 3×3 随机旋转矩阵，返回形状 (batch_size, 3, 3)。
    """
    if device is None:
        device = torch.device("cpu")

    # 1. 生成 4 维独立标准正态
    q = torch.randn(batch_size, 4, device=device, dtype=dtype)
    # 2. 归一化为单位四元数
    q = q / q.norm(dim=1, keepdim=True)

    # 拆分分量
    w, x, y, z = q.unbind(dim=1)

    # 3. 四元数 → 旋转矩阵
    R = torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ), dim=-1).reshape(batch_size, 3, 3)

    return R


# 示例用法
if __name__ == "__main__":
    vectors = {
        "cat": torch.randn(256),
        "dog": torch.randn(256),
        "bird": torch.randn(256),
    }

    path = "./mydb/animals.lmdb"

    # 保存向量
    save_vectors_to_lmdb(vectors, path)

    # 读取向量
    vec = load_vector_from_lmdb("dog", path)
    print("Loaded vector for 'dog':", vec.shape if vec is not None else "Not Found")
    
