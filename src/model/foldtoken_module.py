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

class GeoFeat(nn.Module):
    def __init__(self, geo_layer, num_hidden, virtual_atom_num, dropout=0.0):
        super(GeoFeat, self).__init__()
        self.__dict__.update(locals())
        self.virtual_atom = nn.Linear(num_hidden, virtual_atom_num*3)
        self.virtual_direct = nn.Linear(num_hidden, virtual_atom_num*3)
        self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+32, num_hidden, num_hidden, dropout)
        self.MergeEG = nn.Linear(num_hidden+num_hidden, num_hidden)

    def forward(self, h_V, h_E, attn_mask, trans, rot):
        B, L = h_V.shape[:2]
        V_local = self.virtual_atom(h_V)
        V_edge = self.virtual_direct(h_E)
        Ks = torch.cat([V_edge,V_local[:,:,None].repeat(1,1,L,1)], dim=-1)
        Ks = Ks.view(*V_edge.shape[:-1], -1, 3)
        
        Qt = (rot[...,None,:,:]@Ks[...,None])[...,0]+trans[...,None,:]
        
        RK = rot@rearrange(V_local.view(*V_local.shape[:-1],1,-1,3), 'b i j k d -> b i j d k')
        QRK = V_local.view(*V_local.shape[:-1],1,-1,3)[...,None,:]@rearrange(RK, 'b i j k d -> b i j d k')[...,None]
        QRK = QRK[...,0,0]
        
        
        D = rbf(trans.norm(dim=-1), 0, 50, 16)
        H = torch.cat([Ks.reshape(B,L,L,-1), Qt.reshape(B,L,L,-1), rot.reshape(B,L,L,-1), D, QRK], dim=-1)
        G_e = self.we_condition(H.view(B*L*L,-1)).view(B,L,L,-1)
        h_E = self.MergeEG(torch.cat([h_E, G_e], dim=-1))
        return h_E*attn_mask[...,None]

class GeoAttn(nn.Module):
    def __init__(self, attn_layer, num_hidden, num_V, num_E, dropout=0.0):
        super(GeoAttn, self).__init__()
        self.__dict__.update(locals())
        self.num_heads = 4
        self.W_V = nn.Sequential(nn.Linear(num_E, num_hidden),
                                nn.GELU())
                                
        self.Bias = nn.Sequential(
                                nn.Linear(2*num_V+num_E, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,self.num_heads))
        self.W_O = nn.Linear(num_hidden, num_V, bias=False)
        self.gate = nn.Linear(num_hidden, num_V)


    def forward(self, h_V, h_E, attn_mask):
        h_V_skip = h_V
        L = h_V.shape[1]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        w = self.Bias(torch.cat([h_V[:,:,None].repeat(1,1,L,1), h_E, h_V[:,None,:].repeat(1,L,1,1)], dim=-1))
        attend_logits = w/np.sqrt(d)
        V = self.W_V(h_E)
        attend = F.softmax(attend_logits-(~attn_mask[...,None])*999999999, dim=-2)
        h_V = attend[...,None]*V.reshape(*V.shape[:-1],4,-1)
        h_V = h_V.sum(-3).reshape(*h_V.shape[:2],-1)
        h_V_gate = F.sigmoid(self.gate(h_V))
        dh = self.W_O(h_V)*h_V_gate
        h_V = h_V_skip + dh
        return h_V


class UpdateNode(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden),
            nn.BatchNorm1d(num_hidden)
        )
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden))
    
    def forward(self, h_V, attn_mask):
        B, L, d = h_V.shape
        dh = self.dense(h_V.view(B*L,d)).view(B,L,d)
        h_V = h_V + dh

        # # ============== global attn - virtual frame
        select = attn_mask.sum(dim=-1)>0
        c_V = (h_V*select[...,None]).sum(dim=1)/select.sum(dim=1, keepdim=True)

        h_V = h_V * F.sigmoid(self.V_MLP_g(c_V))[:,None]
        return h_V





    
class StructureSimEncoder(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 encoder_layer,
                 hidden_dim, 
                 dropout=0):
        """ Graph labeling network """
        super(StructureSimEncoder, self).__init__()
        self.__dict__.update(locals())


        self.node_embedding = build_MLP(2, 57, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, 85, hidden_dim, hidden_dim)

        self.interface_layers = FrameFormer(geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 hidden_dim, 
                 dropout=dropout)
        self.proj1 = nn.Linear(hidden_dim, 1280)
        self.encoder_layers=TransformerStack(
            1280, 20, 1, encoder_layer-1, scale_residue=False, n_layers_geom=0
        )
        self.proj2 = nn.Linear(1280, hidden_dim)
        



    def forward(self, V, E, E_trans, E_rots, attn_mask):
        B, L, d = V.shape
        h_V = self.node_embedding(V.view(-1,d)).view(B,L,-1)
        h_E = self.edge_embedding(E.view(B*L*L,-1)).view(B,L,L,-1)
        h_V, h_E = self.interface_layers(h_V, h_E, E_trans, E_rots, attn_mask)
        h_V = self.proj1(h_V)
        h_V = self.encoder_layers(h_V, attn_mask)
        h_V = self.proj2(h_V)
        return h_V

class StructureSimEncoder2(nn.Module):
    def __init__(self, 
                 encoder_layer,
                 hidden_dim, 
                 input_node_dim=16):
        """ Graph labeling network """
        super(StructureSimEncoder2, self).__init__()
        self.__dict__.update(locals())


        self.node_embedding = build_MLP(2, input_node_dim, hidden_dim, 1280)
        # self.edge_embedding = build_MLP(2, 85, hidden_dim, 1280)
        
        self.encoder_layers=TransformerStack(
            1280, 20, 1, encoder_layer, scale_residue=False, n_layers_geom=0, is_geo_attn=True
        )
        self.proj = nn.Linear(1280, hidden_dim)
        

    def forward(self, position, V,  blocks, attn_mask):
        B, L, _ = V.shape
        # h_V = self.node_embedding(self.type_embedding(types).reshape(B,L,-1))
        h_V = self.node_embedding(V.reshape(B*L,-1)).reshape(B,L,-1)
        ## TO DO 计算图
        h_V = self.encoder_layers(position, h_V, attn_mask, blocks=blocks)
        h_V = self.proj(h_V)
        return h_V
    
    

class StructureSimEncoder3(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 encoder_layer,
                 hidden_dim, 
                 dropout=0):
        """ Graph labeling network """
        super(StructureSimEncoder3, self).__init__()
        self.__dict__.update(locals())


        self.node_embedding = build_MLP(2, 57, hidden_dim, 1280)
        self.edge_embedding = build_MLP(2, 85, hidden_dim, 1280)
        self.encoder_layers=TransformerStack(
            1280, 20, 1, encoder_layer, scale_residue=False, n_layers_geom=0
        )
        self.proj2 = nn.Linear(1280, hidden_dim)
        

    def forward(self, V, E, E_trans, E_rots, attn_mask):
        B, L, d = V.shape
        h_V = self.node_embedding(V.view(-1,d)).view(B,L,-1)
        h_E = self.edge_embedding(E.view(B*L*L,-1)).view(B,L,L,-1)
        h_V = self.encoder_layers(h_V, attn_mask, kv_mat=h_E)
        h_V = self.proj2(h_V)
        return h_V


class StructureDecoder(nn.Module):
    def __init__(
        self,
        d_model=1280,
        n_heads=20,
        n_layers=8,
        
    ):
        super().__init__()
        self.decoder_channels = d_model
        self.vq_enc = nn.Linear(128, d_model)
        self.decoder_stack = TransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0, is_geo_attn=False
        )
        self.pred_head_struct = nn.Linear(d_model, 3*5)
        

    def forward(
        self,
        position,
        z_q,
        attention_mask = None,
    ): 
        x = self.vq_enc(z_q)
        x = self.decoder_stack(position, x, attn_mask=attention_mask)
        B, L, _ = x.shape
        pred_x = self.pred_head_struct(x).view(B, L, -1, 3)
        return pred_x, x