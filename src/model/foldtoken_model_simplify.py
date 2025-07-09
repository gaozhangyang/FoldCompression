import torch
import torch.nn as nn
from src.model.foldtoken_module import StructureDecoder,StructureSimEncoder2
from src.model.chroma.struct_loss import ReconstructionLosses
from src.model.chroma.transforms import transform_cbach_to_sbatch
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Type, TypeVar
from bionemo.llm.utils import iomixin_utils as iom
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
)
from bionemo.llm.api import MegatronLossType
from megatron.core.transformer.enums import ModelType
from torch import Tensor
from megatron.core.transformer.enums import AttnBackend
from src.model.transformer_stack import TransformerStack

class FoldCompressionModel(LanguageModule):
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    def __init__(self, config: TransformerConfig, 
                 enc_layers, 
                 dec_layers, 
                 hidden_dim):
        """ Graph labeling network """
        super(FoldCompressionModel, self).__init__(config)
        self.config: TransformerConfig = config
        self.model_type = ModelType.encoder_or_decoder
        self.struct_encoder = StructureSimEncoder2(3, 3, 3, 3, enc_layers, hidden_dim, dropout=0.0)
        
        # self.seq_head = nn.Linear(hidden_dim, 21)
        self.struct_decoder = StructureDecoder(n_layers = dec_layers)
        
        
    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
        
    def compute_loss(self, pred_X, chain, X_true, S_pred, S_true, loss_mask, prefix_num): 
        _, pred_X_batch, _ = transform_cbach_to_sbatch(chain, pred_X)
        _, loss_mask_batch, _ = transform_cbach_to_sbatch(chain, loss_mask[...,None,None])
        C_batch, X_true_batch, _ = transform_cbach_to_sbatch(chain, X_true)
        B,L = C_batch.shape
        X_true_batch = X_true_batch.reshape(B,L,-1,3)
        pred_X_batch = pred_X_batch.reshape(B,L,-1,3)
        mask_batch = torch.isnan(X_true_batch.sum(dim=(-2,-1)))
        C_batch[mask_batch]=-1
        pred_X_batch[mask_batch]=0
        X_true_batch[mask_batch]=0

        results = self.struct_loss(pred_X_batch[:,prefix_num:], X_true_batch[:,prefix_num:], C_batch[:,prefix_num:])
        
        # B,L,d = S_pred.shape
        # seq_loss = F.cross_entropy(S_pred.reshape(B*L,d),S_true.reshape(B*L),reduction='none').reshape(B,L)
        # seq_loss = (seq_loss*loss_mask).sum()/loss_mask.sum()
                    
        
        out = {}
        loss = 0
        for key in ['batch_global_mse', 'batch_fragment_mse', 'batch_pair_mse', 'batch_neighborhood_mse', 'batch_distance_mse', 'batch_hb_local', 'batch_hb_nonlocal', 'batch_hb_contact_order']:
            if results.get(key):
                loss += results[key]
                out.update({key: results[key]})
        out.update({'loss': loss})
        # out.update({'seq_loss': seq_loss})
        
        return out
                                                      
    def forward(self, position, seq_ids, V, blocks, attn_mask, temperature):
        h_V = self.struct_encoder(position, V, blocks, attn_mask)
        eps = torch.finfo(h_V.dtype).eps
        h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True)+eps)

        select = torch.rand_like(seq_ids.float())>temperature
        # h_V[(seq_ids!=34)&select]=0
        h_V[(seq_ids!=34)]=0
        predX = self.struct_decoder(position, h_V, attn_mask)
        return predX,  h_V

class FoldCompressionFMModel(LanguageModule):
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    def __init__(self, config: TransformerConfig, 
                 enc_layers, 
                 dec_layers, 
                 hidden_dim):
        """ Graph labeling network """
        super(FoldCompressionFMModel, self).__init__(config)
        self.config: TransformerConfig = config
        self.model_type = ModelType.encoder_or_decoder
        self.struct_encoder = StructureSimEncoder2(3, 3, 3, 3, enc_layers, hidden_dim, dropout=0.0)
        
        self.xt_proj = nn.Linear(15, 128)
        self.enc_proj = nn.Linear(hidden_dim, 1280)
        self.struct_decoder = TransformerStack(
            1280, 20, 1, dec_layers, scale_residue=False, n_layers_geom=0, is_geo_attn=False, geo_attn_dim=25
        )
        self.pred_head_struct = nn.Linear(1280, 3*5)
        self.time_embed = nn.Linear(1, hidden_dim)
        
        
    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
        
                                                      
    def forward(self, position, seq_ids, V, blocks, attn_mask, xt, t, mode='train'):
        if mode == 'train':
            h_V = self.struct_encoder(position, V, blocks, attn_mask)
            eps = torch.finfo(h_V.dtype).eps
            h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True)+eps)

            prefix_num = (seq_ids==34).sum(dim=-1)[0]
            B, L, K, _ = blocks.shape
            h_V[:, prefix_num:]=self.time_embed(t[:,0])+self.xt_proj(xt[:,:,:5].reshape(B,L,-1)[:,prefix_num:])

            
            h_V = self.enc_proj(h_V)
            h_V = self.struct_decoder(position, h_V, attn_mask=attn_mask)
            predV = self.pred_head_struct(h_V)
            # B,L,_ = h_V.shape
            # predX = xt[:,:,:5]-t*predV.reshape(B,L,-1,3)
            # return predX
            return predV
        else:
            predX = self.sampling(position, seq_ids, V, blocks, attn_mask, 100)
            return predX
    
    def sampling(self, position, seq_ids, V, blocks, attn_mask, n_steps=100):
        """
        使用欧拉方法从 t=1 到 t=0 生成样本。
        model: 训练好的 FlowMatchNet
        num_samples: 要生成的样本数量
        n_steps: 时间离散步数
        返回重构后的 x0: [num_samples, dim]
        """
        device = position.device
        dtype = V.dtype
        B, L, K, _ = blocks.shape
        # 初始噪声 z ~ N(0, I)
        xt = torch.randn((B,L,5,3), device=device, dtype=dtype)
        # 时间步序列，从 1.0 到 0.0
        t_seq = torch.linspace(1.0, 0.0, n_steps, device=device, dtype=dtype)
        dt = t_seq[0] - t_seq[1]
        
        h_V = self.struct_encoder(position, V, blocks, attn_mask)
        eps = torch.finfo(h_V.dtype).eps
        h_V_enc = h_V / (torch.norm(h_V, dim=-1, keepdim=True)+eps)

        prefix_num = (seq_ids==34).sum(dim=-1)[0]

        for t in t_seq[:-1]:
            h_V_enc[:, prefix_num:]=self.time_embed(t[None, None, None])+self.xt_proj(xt.reshape(B,L,-1)[:,prefix_num:])
            h_V = self.enc_proj(h_V_enc)
            h_V = self.struct_decoder(position, h_V, attn_mask=attn_mask)
            v = self.pred_head_struct(h_V)
            v = v.reshape(B,L,-1,3)
            xt = xt + v * (-dt)  # 反向积分：x_{t - dt} = x_t + v * (-dt)
        return xt
        

FoldCompModelT = TypeVar("FoldCompModelT", bound=FoldCompressionModel)

from dataclasses import dataclass
from typing import Type, Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from src.model.foldtoken_module import StructureDecoder, StructureSimEncoder2


from task.loss import CustomLossWithReduction

@dataclass
class FoldCompressionConfig(TransformerConfig, iom.IOMixinWithGettersSetters
):
    """
    Minimal configuration for FoldCompressionModel.

    Attributes:
        model_cls: the model class to instantiate.
        enc_layers: number of layers in the encoder.
        dec_layers: number of layers in the decoder.
        hidden_dim: hidden dimension size.
        dropout: dropout rate.
        max_seq_length: optional maximum sequence length for position embeddings.
    """
    model_cls: Type[FoldCompressionModel] = FoldCompressionModel
    enc_layers: int = 8
    dec_layers: int = 8
    hidden_dim: int = 1280
    dropout: float = 0.0
    max_seq_length: Optional[int] = None
    loss_reduction_class: Type[MegatronLossType] = CustomLossWithReduction
    attention_backend: AttnBackend = AttnBackend.auto
    calculate_per_token_loss: bool = False
    barrier_with_L1_time: bool = False
    fp8: Optional[str] = None



    def configure_model(self) -> FoldCompressionModel:
        """
        Instantiate the FoldCompressionModel with this configuration.
        """
        # Build a TransformerConfig with only the essential fields
        base_cfg = TransformerConfig(
            hidden_size=self.hidden_dim,
            num_attention_heads=max(1, self.hidden_dim // 64),
            num_layers=max(self.enc_layers, self.dec_layers),
            sequence_length=self.max_seq_length or 1024,
            hidden_dropout=self.dropout,
            attention_dropout=self.dropout,
        )
        # Instantiate the model
        model = self.model_cls(
            base_cfg,
            enc_layers=self.enc_layers,
            dec_layers=self.dec_layers,
            hidden_dim=self.hidden_dim
        )
        return model
    
    def get_loss_reduction_class(self) -> Type[MegatronLossType]:  # noqa: D102
        # You could optionally return a different loss reduction class here based on the config settings.
        return self.loss_reduction_class


