# model_interface.py
# User-customizable subclass where you implement model-specific logic and steps

from typing import Iterator, Optional, Dict
import torch
from bionemo.llm.api import MegatronModelType, MegatronLossType
from src.interface.model_interface_base import ModelInterfaceBase
from src.model.foldtoken_model_simplify import FoldCompressionConfig, FoldCompressionFMModel,LatentFMModel
from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from typing import Iterator, Optional, Dict, Any
from .loss import compute_custom_loss
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from megatron.core.optimizer import OptimizerConfig
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler



class BionemoLightningModule(
    ModelInterfaceBase[MegatronModelType, MegatronLossType]
):
    """User implementation: override only these methods to define your model."""
    def __init__(
        self,
        model_transform: Optional[Any] = None,
        configure_init_model_parallel: bool = False,
        enc_layers: int = 12,
        dec_layers: int = 12,
        hidden_dim: int = 128,
        prefix_len: int = 6,
        warmup_steps: int = 1000,
        lr: float = 1e-4,
        scheduler_num_steps: int = 10000,
        custom_checkpoint_path: Optional[str] = None,
        **model_construct_args,
    ) -> None:
        """Pass through all initialization args to the base class."""
        super().__init__(
            model_transform=model_transform,
            configure_init_model_parallel=configure_init_model_parallel,
            **model_construct_args,
        )
        self.save_hyperparameters()
        optimizer = self.set_optimizer()
        self.optim = optimizer
        self.optim.connect(self)
        self.config = self.set_config()
        self.loss_reduction_class = self.config.get_loss_reduction_class()
        
        
    def set_config(self):
        self.config = FoldCompressionConfig(
            enc_layers=self.hparams.enc_layers,
            dec_layers=self.hparams.dec_layers,
            hidden_dim=self.hparams.hidden_dim,
            dropout=0.0,
            max_seq_length=1024,
            num_attention_heads=1,
            num_layers=1,
        )
        return self.config

    def configure_model(self) -> None:
        """Instantiate the FoldCompressionModel and assign to self.module"""
        self.module = LatentFMModel(
            self.config,
            self.hparams.enc_layers,
            self.hparams.dec_layers,
            self.hparams.hidden_dim,
        )
        if self.hparams.custom_checkpoint_path != "":
            self.load_from_torch_ckpt(self.hparams.custom_checkpoint_path)

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
        return get_batch_on_this_context_parallel_rank(_batch)

    def forward_step(self, batch: Dict, mode='train') -> Dict:
        """Core forward: build attention mask, compute features, and run the model."""
        xt, t, v = self.sample_flow_batch(batch['value'])
          
        B, L = batch['value'].shape[:2]
        
        predX = self.module(
                xt,
                t,
                mode
            )
        return {'predX': predX,  'v':v}
    


    def training_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        """Training step: set prefix length and run forward_step."""
        batch['prefix_len'] = self.hparams.prefix_len
        outputs = self.forward_step(batch)
        loss, results = compute_custom_loss(outputs, batch, 'train')
        if self.is_on_logging_device():
            self.log("train_loss", results['loss'], on_step=True, on_epoch=True, prog_bar=True)
            for key, val in results.items():
                self.log("train_"+key, val, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def validation_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        """Validation step: set prefix length, eval mode, and run forward_step without gradient."""
        
        batch['prefix_len'] = self.hparams.prefix_len
        with torch.no_grad():
            self.module.eval()
            outputs = self.forward_step(batch, 'train')
            if self.is_on_logging_device():
                loss, results = compute_custom_loss(outputs, batch, 'train')
                for key, val in results.items():
                    if key != 'loss':
                        self.log("val_"+key, val, on_step=False, on_epoch=True, prog_bar=True)
            return outputs
    

    def predict_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Optional[Dict]:
        """Predict step alias to forward_step for inference."""
        if not batch:
            return None
        return self.forward_step(batch)
    


    def set_optimizer(self):
        optimizer = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=self.hparams.lr,
                optimizer="adam",
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=self.hparams.warmup_steps,
                max_steps=self.hparams.scheduler_num_steps,
                max_lr=self.hparams.lr,
                min_lr=0.0,
                anneal_percentage=0.10,
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
    
    def sample_flow_batch(self, x0):
        """
        给定原始样本 x0，随机采样 z ~ N(0,I)，采样 t ~ Uniform(0,1)，
        构造 x_t = (1 - t) * x0 + t * z，目标速度 v = z - x0。
        返回 x_t, t, v。
        """
        # x0: [B, dim]
        z = torch.randn_like(x0)
        t = torch.rand(x0.shape[0], 1, 1, device=x0.device)
        xt = (1 - t) * x0 + t * z
        v = z - x0
        return xt, t, v
    
    def init_process_group_if_needed(self, backend="gloo"):
        import torch.distributed as dist
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'


        if not dist.is_initialized():
            # 单进程用 gloo 足够
            dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0)

    def extract_plain_state(self, ckpt_dir: str, output_path: str):
        from megatron.core.dist_checkpointing import load_plain_tensors
        # ⚠️ 一定要先初始化分布式 group
        self.init_process_group_if_needed(backend="gloo")
        # 加载 shard checkpoint 自动合并
        state_dict = load_plain_tensors(ckpt_dir)
        torch.save(state_dict, output_path)
        print(f"✔️ Saved merged checkpoint at: {output_path}")
        return state_dict