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
        geo_attn_dim=16,
        scale=100
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.scale = scale

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)
        if is_geo_attn:
            # self.edge_embed = nn.Linear(geo_attn_dim, 64, bias=False)
            self.geo_key = nn.Linear(3, n_heads, bias=False)
            self.geo_query = nn.Linear(3, n_heads, bias=False)


    def _apply_rotary(self, position, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(position, q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, position, x, attention_mask=None, blocks=None, atom_mask=None):
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
            dtype = blocks.dtype
            scale = self.scale # 这个缩放非常有必要，防止因为精度范围溢出导致的旋转/平移不变性失效
            struct_mask = ~((blocks==0).all(dim=(-1,-2))) # True where structure is valid
            X = blocks/scale # [bacth, L, 4, 3]
            X_bar = X.mean(dim=-2, keepdims=True)
            B = X - X_bar
            # self.geo_query(X.permute(0,1,3,2)).permute(0,1,3,2)
            # B = B/(torch.norm(B, dim=-1)[...,None]+1e-6)
            v = (B * X_bar).sum(dim=-1, keepdims=True)
            Q = torch.cat([B, -v], dim=-1)
            K = torch.cat([X, torch.ones_like(v)], dim=-1)
            Q_BHLD = self.geo_query(Q.permute(0,1,3,2)).permute(0,3,1,2)
            K_BHLD = self.geo_key(K.permute(0,1,3,2)).permute(0,3,1,2)
            
            query_BHLD = torch.cat([query_BHLD, Q_BHLD*struct_mask[:,None,:,None]], dim=-1)
            key_BHLD = torch.cat([key_BHLD, K_BHLD*struct_mask[:,None,:,None]], dim=-1)
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
            out_X = None
            
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
