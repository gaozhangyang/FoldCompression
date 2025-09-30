import torch
import torch.nn as nn
from src.model.module import StructureDecoder,StructureSimEncoder2
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



class FoldCompressionModel(LanguageModule):
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    def __init__(self, config: TransformerConfig, 
                 enc_layers, 
                 dec_layers, 
                 hidden_dim,
                 nn_neighbors):
        """ Graph labeling network """
        super(FoldCompressionModel, self).__init__(config)
        self.config: TransformerConfig = config
        self.model_type = ModelType.encoder_or_decoder
        self.struct_encoder = StructureSimEncoder2( enc_layers, hidden_dim, input_node_dim=nn_neighbors*9, scale=100)
        
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
                                                      
    def forward(self, position, seq_ids, V, blocks, attn_mask, infer_feats=False):
        h_V = self.struct_encoder(position, V, blocks, attn_mask)
        eps = torch.finfo(h_V.dtype).eps
        h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True)+eps)
        if infer_feats:
            return h_V

        # select = torch.rand_like(seq_ids.float())>temperature
        # h_V[(seq_ids!=34)&select]=0
        h_V[(seq_ids!=34)]=0
        predX, h_V = self.struct_decoder(position, h_V, attn_mask)
        
        # h_V = self.struct_refiner(position, h_V, blocks=predX.detach(), attn_mask=attn_mask)
        # B,L,_ = h_V.shape
        # attn = F.softmax(h_V@h_V.permute(0,2,1)-torch.eye(L,L, device=h_V.device, dtype=h_V.dtype)[None], dim=-1)
        # # diffX = (predX[:,None]-predX[:,:,None]).detach()
        # # diffX = diffX / (torch.norm(diffX, dim=-1, keepdim=True)+torch.finfo(diffX.dtype).eps)
        # # shift_X = (attn[...,None,None]*diffX).sum(dim=2).reshape(B,L,-1,3)
        # shift_X =  (attn@(predX.detach().view(B,L,-1)/L)).reshape(B,L,-1,3)
        # predX2 = predX.detach() + shift_X*self.update_gate(h_V).reshape(B,L,-1,1)
        return predX


FoldCompModelT = TypeVar("FoldCompModelT", bound=FoldCompressionModel)

from dataclasses import dataclass
from typing import Type, Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from src.model.module import StructureDecoder, StructureSimEncoder2




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
    from task.loss import CustomLossWithReduction
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

