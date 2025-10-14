import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
from src.model.transformer_stack import TransformerStack



def sparse_to_dense(h_E, neighborhood, L):
    """
    将稀疏表示 (h_E, neighborhood) 转换为密集表示 (E, attn_mask)
    
    参数:
    - h_E: (batch, L, k, dim)  pairwise feature 矩阵
    - neighborhood: (batch, L, k)  邻居索引矩阵
    - L: 序列长度 (581)
    
    返回:
    - E: (batch, L, L, dim)  全连接 pairwise feature 矩阵
    - attn_mask: (batch, L, L)  邻接矩阵，标记有效连接
    """
    batch, _, k, dim = h_E.shape
    
    # 初始化 E 和 attn_mask
    E = torch.zeros((batch, L, L, dim), device=h_E.device)
    attn_mask = torch.zeros((batch, L, L), device=h_E.device, dtype=torch.bool)

    # 使用 torch.gather 进行索引填充
    batch_idx = torch.arange(batch).view(batch, 1, 1).expand(batch, L, k)
    src_idx = torch.arange(L).view(1, L, 1).expand(batch, L, k)
    
    # 填充 E 和 attn_mask
    E[batch_idx, src_idx, neighborhood] = h_E
    attn_mask[batch_idx, src_idx, neighborhood] = 1

    return E, attn_mask


def dense_to_sparse(E, attn_mask, neighborhood):
    """
    将密集表示 (E, attn_mask) 转换回稀疏表示 (h_E, E_idx)
    
    参数:
    - E: (batch, L, L, dim)  全连接 pairwise feature 矩阵
    - attn_mask: (batch, L, L)  邻接矩阵
    - neighborhood: (batch, L, k)  邻居索引矩阵
    
    返回:
    - h_E: (batch, L, k, dim)  稀疏 pairwise feature 矩阵
    - E_idx: (batch, L, k)  邻居索引矩阵 (同 neighborhood)
    """
    batch, L, k = neighborhood.shape
    dim = E.shape[-1]

    batch_idx = torch.arange(batch).view(batch, 1, 1).expand(batch, L, k)
    src_idx = torch.arange(L).view(1, L, 1).expand(batch, L, k)

    # 使用 gather 获取 h_E
    h_E = E[batch_idx, src_idx, neighborhood]
    E_idx = neighborhood  # 直接复用原来的索引

    return h_E, E_idx

def rbf_func(D, num_rbf):
    shape = D.shape
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count, dtype=D.dtype, device=D.device)
    D_mu = D_mu.view([1]*(len(shape))+[-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


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


class FoldRepInputLayer(nn.Module):
    def __init__(self, structure_dim, output_dim=1280, modality=['structure', 'sequence']):
        super(FoldRepInputLayer, self).__init__()
        self.modality = modality
        if 'structure' in modality:
            self.struct_embedding = build_MLP(2, structure_dim, output_dim, output_dim)
        if 'sequence' in modality:
            self.seq_embedding = nn.Embedding(35, output_dim)
            
        
    def forward(self, struct_x, seq_x):
        out_x = 0
        if 'structure' in self.modality:
            B, L, _ = struct_x.shape
            out_x += self.struct_embedding(struct_x.reshape(B*L,-1)).reshape(B,L,-1)
        if 'sequence' in self.modality:
            out_x += self.seq_embedding(seq_x)
        return out_x


class FoldRepEncoder(nn.Module):
    def __init__(self, 
                 encoder_layer,
                 d_model=1280,
                 n_heads=20,
                 input_node_dim=9,
                 scale=100):
        """ Graph labeling network """
        super(FoldRepEncoder, self).__init__()
        self.__dict__.update(locals())


        # self.node_embedding = build_MLP(2, input_node_dim, hidden_dim, 1280)
        # self.edge_embedding = build_MLP(2, 85, hidden_dim, 1280)
        
        self.encoder_layers=TransformerStack(
            d_model, n_heads, 1, encoder_layer, scale_residue=False, n_layers_geom=0, is_geo_attn=True, scale=scale
        )
        # self.proj = nn.Linear(1280, hidden_dim)
        

    def forward(self, position, h_V,  blocks, attn_mask, input_modality=['structure', 'sequence']):
        B, L, _ = h_V.shape
        # h_V = self.node_embedding(self.type_embedding(types).reshape(B,L,-1))
        # h_V = self.node_embedding(V.reshape(B*L,-1)).reshape(B,L,-1)
        if 'structure' not in input_modality:
            blocks = torch.zeros_like(blocks)
        ## TO DO 计算图
        h_V = self.encoder_layers(position, h_V, attn_mask, blocks=blocks)
        # h_V = self.proj(h_V)
        return h_V
    


class FoldRepDecoder(nn.Module):
    def __init__(
        self,
        d_model=1280,
        n_heads=20,
        n_layers=8,
        
    ):
        super().__init__()
        self.decoder_channels = d_model
        # self.vq_enc = nn.Linear(128, d_model)
        self.decoder_stack = TransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0, is_geo_attn=False, scale=100
        )
        # self.pred_head_struct = nn.Linear(d_model, 3*5)
        

    def forward(
        self,
        position,
        x,
        attention_mask = None,
    ): 
        # x = self.vq_enc(z_q)
        x = self.decoder_stack(position, x, attn_mask=attention_mask)
        return x
        # B, L, _ = x.shape
        # pred_x = self.pred_head_struct(x).view(B, L, -1, 3)
        # return pred_x, x

class FoldRepModalityHead(nn.Module):
    def __init__(self, d_model, modality=['structure', 'sequence']):
        super(FoldRepModalityHead, self).__init__()
        self.d_model = d_model
        self.modality = modality
        if 'structure' in modality:
            self.struct_head = nn.Linear(d_model, 3*5)
        if 'sequence' in modality:
            self.seq_head = nn.Linear(d_model, 35)

        
    def forward(self, x):
        out = {}
        if 'structure' in self.modality:
            struct_x = self.struct_head(x)
            out['struct_x'] = struct_x
        if 'sequence' in self.modality:
            seq_x = self.seq_head(x)
            out['seq_x'] = seq_x
        return out