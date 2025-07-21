import torch
import os, sys
from megatron.core.transformer.enums import AttnBackend
sys.path.append('/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression')
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"


from src.model.foldtoken_model_simplify import FoldCompressionConfig, FoldCompressionModel



enc_layers = 8
dec_layers = 1
hidden_dim = 128

config = FoldCompressionConfig(
    enc_layers=enc_layers,
    dec_layers=dec_layers,
    hidden_dim=hidden_dim,
    dropout=0.0,
    max_seq_length=1024,
    num_attention_heads=1,
    num_layers=1,
    attention_backend = AttnBackend.unfused
)
        
model = FoldCompressionModel(
    config,
    enc_layers,
    dec_layers,
    hidden_dim,
)

ckpt_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3/checkpoints/epoch=0-step=999999-consumed_samples=64192032.0-last.pt"  # 替换为实际的模型检查点路径

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
ckpt = {key.replace('module.',''): val for key, val in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
print(f"✅ 模型加载成功：{ckpt_path}")
model.eval()

B = 32
length = 155
embed_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/task/h_V.pt"
position = torch.cat([torch.arange(31,-1,-1), torch.arange(0,length)+32])
position = position[None].repeat(B,1)
attn_mask = torch.ones((B, length+32, length+32), dtype=torch.bool)>0
h_V = torch.load(embed_path).float().cpu()

predX = model.struct_decoder(position, h_V, attn_mask)


