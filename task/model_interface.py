# model_interface.py
# User-customizable subclass where you implement model-specific logic and steps

from typing import Iterator, Optional, Dict
import torch
from bionemo.llm.api import MegatronModelType, MegatronLossType
from src.interface.model_interface_base import ModelInterfaceBase
from src.model.foldtoken_model_simplify import FoldCompressionConfig, FoldCompressionModel
from bionemo.llm.model.biobert.lightning import get_batch_on_this_context_parallel_rank
from typing import Iterator, Optional, Dict, Any
from .loss import compute_custom_loss
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from megatron.core.optimizer import OptimizerConfig
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from src.data.protein import Protein
from torch.cuda import empty_cache

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
        infer_feats: int = 0,
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
        self.module = FoldCompressionModel(
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

        B, L = batch['blocks'].shape[:2]
        blocks = batch['blocks']
        M = blocks.mean(dim=-2, keepdims=True)
        base = blocks - M
        eps = torch.finfo(base.dtype).eps
        base = base / (torch.norm(base, dim=-1, keepdim=True) + eps)
        V = torch.einsum('bqex,bqcx->bqec', base, blocks).reshape(B, L, -1)

        # all_steps = len(self.trainer.datamodule.train_dataloader())
        all_steps = 20000
        temperature = torch.clamp(torch.tensor(1-self.global_step/all_steps), 0, 1.0).to(batch['position'].device)
        
        predX, h_V = self.module(
            batch['position'],
            batch['seq_ids'],
            V,
            batch['blocks'],
            attn_mask,
            temperature
        )
        return {'predX': predX, 'mask': attn_mask, 'h_V':h_V}

    def training_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        """Training step: set prefix length and run forward_step."""
        batch['prefix_len'] = self.hparams.prefix_len
        outputs = self.forward_step(batch)
        loss, results = compute_custom_loss(outputs, batch)
        
        # idx = 2
        for idx in range(8):
            true_X = batch['coords'][:,:,:5]
            mask0 = true_X.sum(dim=(-2,-1))!=0
            mask = (batch['data_id']>0)&mask0
            pred_X = outputs['predX']
            X = pred_X[idx][mask[idx]][None][:,:,[0,1,2,4]]
            C = torch.ones_like(X)[:,:,0,0].long()
            protein_pred = Protein.from_XCS(X, C, C)
            
            X = true_X[idx][mask[idx]][None][:,:,[0,1,2,4]]
            protein_true = Protein.from_XCS(X, C, C)
            
            protein_pred.to(f'/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/sample{idx}_pred.pdb')
            protein_true.to(f'/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/sample{idx}_true.pdb')
        
        if self.is_on_logging_device():
            self.log("train_loss", results['loss'], on_step=True, on_epoch=True, prog_bar=True)
            for key, val in results.items():
                self.log("train_"+key, val, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def validation_step(self, batch: Dict, batch_idx: Optional[int] = None) -> Dict:
        """Validation step: set prefix length, eval mode, and run forward_step without gradient."""
        if self.hparams.infer_feats:
            batch['prefix_len'] = self.hparams.prefix_len
            with torch.no_grad():
                self.module.eval()
                outputs = self.forward_step(batch, infer_feats=True)
                h_V = outputs['h_V'][:,:self.hparams.prefix_len]
                names = batch['names'][0]
                save_vectors_to_lmdb({name: h_V[i, :] for i, name in enumerate(names)}, f"/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3_infer/compression_data.lmdb")
                empty_cache()
            # return outputs
            return h_V.mean()
        else:
            batch['prefix_len'] = self.hparams.prefix_len
            with torch.no_grad():
                self.module.eval()
                outputs = self.forward_step(batch)
                if self.is_on_logging_device():
                    loss, results = compute_custom_loss(outputs, batch)
                    for key, val in results.items():
                        if key != 'loss':
                            self.log("val_"+key, val, on_step=False, on_epoch=True, prog_bar=True)
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
                lr=self.hparams.lr,
                optimizer="adam",
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
                clip_grad=1.0
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
    
