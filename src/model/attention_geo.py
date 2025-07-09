import functools

import einops
import torch
import torch.nn.functional as F
from torch import nn

from src.model.rotary import RotaryEmbedding

def compute_rotation_weighted_pca(X, w):
    # X: [B, N, 3],  w: [B, N]
    B, N, _ = X.shape
    w_sum = w.sum(dim=1, keepdim=True)             # [B,1]
    # 1. 加权去中心
    mu = (w.unsqueeze(-1) * X).sum(dim=1, keepdim=True) / w_sum[...,None]  # [B,1,3]
    Xc = X - mu                                      # [B,N,3]
    # 2. 加权协方差
    Xcw = Xc * w.unsqueeze(-1)                      # [B,N,3]
    C = Xc.transpose(1,2) @ Xcw / w_sum.unsqueeze(-1)  # [B,3,3]
    # 3. SVD
    U, S, Vt = torch.svd(C)
    # 4. 保证正定旋转
    det = torch.det(U @ Vt.transpose(1,2))
    D = torch.eye(3, device=X.device).unsqueeze(0).repeat(B,1,1)
    D[:,2,2] = det
    R = U @ D @ Vt.transpose(1,2)
    return R


def build_MLP(n_layers,dim_in, dim_hid, dim_out, dropout = 0.0, activation=nn.ReLU, normalize=True):
    if normalize:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.BatchNorm1d(dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    else:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        if normalize:
            layers.append(nn.BatchNorm1d(dim_hid))
        layers.append(nn.Dropout(dropout))
        layers.append(activation())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
        is_geo_attn=False,
        geo_attn_dim=16
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)
        if is_geo_attn:
            self.edge_embed = nn.Linear(geo_attn_dim, 64, bias=False)


    def _apply_rotary(self, position, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(position, q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, position, x, attention_mask=None, blocks=None):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(position, query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        # Where True, enable participation in attention.
        # mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
        mask_BLL = attention_mask
        mask_BHLL = mask_BLL.unsqueeze(1)
        # torch.cuda.memory._record_memory_history() 
        # count = 0
        B, H, L, D = query_BHLD.shape
        if blocks is not None:
            M = blocks.mean(dim=-2, keepdims=True)
            base_BLK3 = blocks - M
            eps = torch.finfo(blocks.dtype).eps
            base_BLK3 = base_BLK3/(torch.norm(base_BLK3, dim=-1)[...,None]+eps)
            ## ============ uni map, decoupled, checked =============
            length = attention_mask.sum(dim=-1)
            tmp1 = torch.einsum('bhkd, bkcx, bqk->bhqdcx', key_BHLD/length[:,None,:,None], blocks-M, attention_mask.to(blocks.dtype))
            tmp2 = torch.einsum('bhqd, bhqdcx->bhqcx', query_BHLD, tmp1)
            
            context = torch.einsum('bqex, bhqcx->bhqec', base_BLK3, tmp2).reshape(B,H,L,-1) #/length[:,None,:,None]
            context = self.edge_embed(context)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )+context
            out_X = blocks
        else:
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
            out_X = None
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD), out_X


class MultiHeadAttentionSE3(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)
        
        self.pred_trans = nn.Linear(d_model, 3)
        self.pred_rots = nn.Linear(d_model, 9)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        
        
        
        # query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        # Where True, enable participation in attention.
        mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
        mask_BHLL = mask_BLL.unsqueeze(1)

        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD, mask_BHLL
        )
        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)
