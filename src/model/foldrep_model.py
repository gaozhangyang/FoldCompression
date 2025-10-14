# Standard library imports
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
)

# Third-party imports
import torch
import torch.nn as nn
from torch import Tensor

# Megatron imports
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnBackend, ModelType

# Nemo imports
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
)
from nemo.utils import logging

# BioNeMo imports
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from bionemo.llm.utils import iomixin_utils as iom

# Local imports
from src.model.chroma.struct_loss import ReconstructionLosses
from src.model.module import FoldRepDecoder, FoldRepEncoder, FoldRepInputLayer, FoldRepModalityHead
from src.data.omni_dataset import batched_topk_neighbors_3d, index_along_len_tad


class FoldRepModel(LanguageModule):
    """Graph labeling network for protein structure compression."""
    
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    
    def __init__(self, config: TransformerConfig):
        """Initialize the FoldRepModel.
        
        Args:
            config: Transformer configuration
            enc_layers: Number of encoder layers
            dec_layers: Number of decoder layers
            hidden_dim: Hidden dimension size
            nn_neighbors: Number of nearest neighbors
            modality: Modality of the input
        """
        super(FoldRepModel, self).__init__(config)
        self.config = config
        # Read all hyperparameters from config to keep constructor minimal
        self.input_modality = getattr(config, 'input_modality', ['structure', 'sequence'])
        self.output_modality = getattr(config, 'output_modality', ['structure', 'sequence'])
        self.model_type = ModelType.encoder_or_decoder
        
        d_model = getattr(config, 'd_model', getattr(config, 'hidden_size', None))
        if d_model is None:
            raise ValueError("Config must define 'd_model' or 'hidden_size' for FoldRepModel")
        hidden_dim = getattr(config, 'hidden_dim', d_model)
        nn_neighbors = getattr(config, 'nn_neighbors', 9)
        prefix_len = getattr(config, 'prefix_len', 6)
        n_heads = getattr(config, 'n_heads', getattr(config, 'num_attention_heads', 1))
        enc_layers = getattr(config, 'enc_layers', getattr(config, 'num_layers', 1))
        dec_layers = getattr(config, 'dec_layers', getattr(config, 'num_layers', 1))

        self.input_layer = FoldRepInputLayer(structure_dim=nn_neighbors*9, output_dim=d_model, modality=self.input_modality)
        
        self.fold_encoder = FoldRepEncoder(
            enc_layers, d_model=d_model, n_heads=n_heads, input_node_dim=nn_neighbors*9, scale=100
        )
        self.proj = nn.Linear(d_model, hidden_dim)
        
        if getattr(config, 'use_dino', False)==1:
            self.dino_proj = nn.Linear(hidden_dim*prefix_len, hidden_dim)
        self.proj_inv = nn.Linear(hidden_dim, d_model)
        
        self.use_dino = getattr(config, 'use_dino', 0)
        if self.use_dino!=3:
            self.fold_decoder = FoldRepDecoder(n_layers=dec_layers, d_model=d_model, n_heads=n_heads)
            
        
        self.modality_head = FoldRepModalityHead(d_model=d_model, modality=self.output_modality)
    
    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func.
        """
        self.input_tensor = input_tensor
    
    def forward(self, position, seq_ids,  blocks, attn_mask, infer_feats=False):
        """Forward pass of the model.
        
        Args:
            position: Position embeddings
            seq_ids: Sequence IDs
            blocks: Block structure
            attn_mask: Attention mask
            infer_feats: Whether to only return features
            
        Returns:
            Predicted coordinates or features
        """
        if 'structure' in self.input_modality:
            if self.use_dino==4:
                isvalid = (attn_mask.sum(dim=-2)>0)
                
            else:
                isvalid = seq_ids!=-1 # 这里看你有bug,因为seqid似乎没有-1,但是之前代码都是这么写的, 暂时不改,怕和ckpt不兼容
            V = self.construct_nn_input(blocks, isvalid, nn_neighbors=getattr(self.config, 'nn_neighbors', 9))
            if self.use_dino==4:
                V = V * isvalid[:,:,None]
        else:
            V = None
            
        h_V = self.input_layer(V, seq_ids)
        h_V = self.fold_encoder(position, h_V, blocks, attn_mask, input_modality=self.input_modality)
        h_V = self.proj(h_V)
        eps = torch.finfo(h_V.dtype).eps
        h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True) + eps)
        
        if self.use_dino == 3: # mlm baseline
            h_V = self.proj_inv(h_V)
            if infer_feats:
                prefix_len = (seq_ids[0]==34).sum()
                return h_V[:,prefix_len:].mean(dim=-2)
        
            out = self.modality_head(h_V)
        else:
            h_V[seq_ids != 34] = 0
            
            if infer_feats==1:
                prefix_len = (seq_ids[0]==34).sum()
                return self.dino_proj(h_V[:,:prefix_len].reshape(h_V.shape[0], -1))
            elif infer_feats==2:
                prefix_len = (seq_ids[0]==34).sum()
                return h_V[:,:prefix_len].reshape(h_V.shape[0], -1)

            h_V = self.proj_inv(h_V)
            h_V = self.fold_decoder(position, h_V, attn_mask)
            out = self.modality_head(h_V)
        
        return out
    
    def construct_nn_input(self, blocks, isvalid, nn_neighbors):
        B, L, H, _ = blocks.shape
        select = batched_topk_neighbors_3d(blocks[:,:,0], isvalid, nn_neighbors)
        M = (blocks).mean(dim=-2, keepdim=True)
        base = blocks - M
        eps = torch.finfo(base.dtype).eps
        base = base / (torch.norm(base, dim=-1, keepdim=True) + eps)
        blocks_neighbors = index_along_len_tad(blocks, select)
        V = torch.einsum('blex,blkcx->blkec', base, blocks_neighbors-M[:,:,None]).reshape(B, L, -1)
        return V
    
    def compute_custom_loss(
        self,
        output: Dict[str, Tensor],
        batch: Dict[str, Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute custom loss for the model.

        Args:
            output: Model output dictionary containing 'predX'
            batch: Batch dictionary containing 'data_id', 'coords', 'prefix_len'
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (loss, loss_details)
        """
        
        
        chain = batch['data_id']
        prefix_len = batch['prefix_len']
        
        out = {'loss': 0}
        if 'structure' in self.output_modality:
            pred_X = output['struct_x']
            X_true = batch['coords'][:, :, :5]
            struct_loss_func = ReconstructionLosses(rmsd_method='symeig', loss_scale=10.0)
            str_loss = self.compute_struct_loss(X_true, pred_X, struct_loss_func, chain, prefix_len)
            out.update(str_loss)
        
        if 'sequence' in self.output_modality:
            pred_S = output['seq_x']
            # 检查是否是MLM模式 (use_dino=3)
            use_dino = batch.get('use_dino', 0)
            prefix_len = batch.get('prefix_len', 0)
            
            if use_dino == 3:
                S_true = batch['seq_ids_ori']
                prefix_mask = S_true==34
                # MLM模式：只在被mask的位置计算loss
                masked_position = batch.get('masked_position', None)
                if masked_position is not None:
                    mask = masked_position & (chain > 0) & (~prefix_mask)  # 被mask的位置且有效
                else:
                    # 如果没有masked_position，回退到原始逻辑
                    mask = (chain > 0) & (~prefix_mask)
            else:
                S_true = batch['seq_ids']
                prefix_mask = S_true==34
                # 非MLM模式：在所有有效位置计算loss
                mask = (chain > 0) & (~prefix_mask)
            
            seq_loss_func = nn.CrossEntropyLoss(reduction='none')
            seq_loss = seq_loss_func(pred_S.permute(0,2,1), S_true)
            seq_loss = seq_loss[mask].mean()
            recovery = pred_S.argmax(dim=-1) == S_true
            recovery = recovery[mask].to(seq_loss.dtype).mean()
            out['seq_loss'] = seq_loss
            out['loss'] += seq_loss
        
        # Optional DINO feature alignment loss provided via batch
        dino_loss = batch.get('dino_loss', None)
        if dino_loss is not None:
            out['dino_loss'] = dino_loss
            out['loss'] = out['loss'] + dino_loss
        
        contrastive_loss = batch.get('contrastive_loss', None)
        if contrastive_loss is not None:
            out['contrastive_loss'] = contrastive_loss
            out['loss'] = out['loss'] + contrastive_loss
        
        return out['loss'], out
        
    @classmethod
    def compute_struct_loss(cls, X_true_batch: Tensor, pred_X_batch: Tensor, 
                     struct_loss: ReconstructionLosses, C_batch: Tensor, 
                     prefix_len: int) -> Dict[str, Tensor]:
        """Compute structural reconstruction loss.
        
        Args:
            X_true_batch: True coordinates
            pred_X_batch: Predicted coordinates
            struct_loss: Structural loss function
            C_batch: Chain information
            prefix_len: Prefix length to skip
            
        Returns:
            Dictionary containing loss components
        """
        B, L = C_batch.shape
        X_true_batch = X_true_batch.reshape(B, L, -1, 3)
        pred_X_batch = pred_X_batch.reshape(B, L, -1, 3)
        
        # Create mask for invalid coordinates
        mask_batch = torch.isnan(X_true_batch.sum(dim=(-2, -1)))
        C_batch[mask_batch] = -1
        pred_X_batch[mask_batch] = 0
        X_true_batch[mask_batch] = 0

        # Compute structural loss
        results = struct_loss(
            pred_X_batch[:, prefix_len:], 
            X_true_batch[:, prefix_len:], 
            C_batch[:, prefix_len:]
        )
        
        # Aggregate loss components
        loss = 0
        loss_keys = ['batch_global_mse', 'batch_fragment_mse', 'batch_pair_mse', 'batch_neighborhood_mse', 'batch_distance_mse']
        for key in loss_keys:
            if results.get(key):
                loss += results[key]
        
        out = {}
        out['batch_global_mse'] = results['batch_global_mse']
        out['loss'] = loss
        return out


class CustomLossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
    """Custom loss reduction class for FoldCompression model."""
    
    def __init__(
        self,
        model,
        validation_step: bool = False,
        val_drop_last: bool = True,
        **loss_kwargs: Any,
    ) -> None:
        """Initialize custom loss reduction.

        Args:
            validation_step: Whether in validation step
            val_drop_last: Whether to drop last incomplete batch in validation
            **loss_kwargs: Additional arguments passed to compute_custom_loss
        """
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        self.loss_kwargs = loss_kwargs
        self.model = model

    def forward(
        self,
        batch: Dict[str, Tensor],
        forward_out: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute and return reduced loss.

        Args:
            batch: Input data dictionary containing 'loss_mask' etc.
            forward_out: Model forward output dictionary containing 'predX'

        Returns:
            Tuple of (reduced_loss, extras_dict)
        """
        # Compute custom loss
        unreduced_loss, _ = self.model.compute_custom_loss(
            forward_out, batch, **self.loss_kwargs
        )

        # Apply micro batch reduction
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            loss_mb = masked_token_loss(unreduced_loss, batch.get('loss_mask', None))
        else:
            # Note: masked_token_loss_context_parallel is commented out in imports
            # This would need to be uncommented if context parallelism is used
            raise NotImplementedError("Context parallelism not implemented")

        # Handle validation step with val_drop_last
        if self.validation_step and not self.val_drop_last:
            num_valid = batch.get('loss_mask', torch.ones_like(unreduced_loss)).sum()
            if loss_mb.isnan():
                if num_valid != 0:
                    raise ValueError("Non-empty input resulted in NaN loss")
                loss_sum_mb = torch.zeros_like(num_valid)
            else:
                loss_sum_mb = num_valid * loss_mb

            buf = torch.cat([
                loss_sum_mb.clone().detach().view(1),
                torch.Tensor([num_valid]).cuda().clone().detach()
            ])
            torch.distributed.all_reduce(
                buf,
                group=parallel_state.get_data_parallel_group(),
                op=torch.distributed.ReduceOp.SUM,
            )
            return loss_mb * cp_size, {'loss_sum_and_microbatch_size': buf}

        # Normal case: average loss across data parallel group
        reduced = average_losses_across_data_parallel_group([loss_mb])
        return loss_mb * cp_size, {'avg': reduced}


# Type variable for model
FoldCompModelT = TypeVar("FoldCompModelT", bound=FoldRepModel)


@dataclass
class FoldRepModelConfig(TransformerConfig, iom.IOMixinWithGettersSetters):
    """Configuration for FoldRepModel.

    Attributes:
        model_cls: The model class to instantiate
        enc_layers: Number of layers in the encoder
        dec_layers: Number of layers in the decoder
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
        max_seq_length: Optional maximum sequence length for position embeddings
        loss_reduction_class: Loss reduction class to use
        attention_backend: Attention backend to use
        calculate_per_token_loss: Whether to calculate per-token loss
        barrier_with_L1_time: Whether to use L1 time barrier
        fp8: FP8 configuration
    """
    model_cls: Type[FoldRepModel] = FoldRepModel
    enc_layers: int = 8
    dec_layers: int = 8
    hidden_dim: int = 1280
    d_model: Optional[int] = None
    n_heads: int = 16
    nn_neighbors: int = 9
    input_modality: List[str] = ("structure", "sequence")
    output_modality: List[str] = ("structure", "sequence")
    prefix_len: int = 6
    dropout: float = 0.0
    max_seq_length: Optional[int] = None
    # loss_reduction_class: Type[MegatronLossType] = CustomLossWithReduction
    attention_backend: AttnBackend = AttnBackend.auto
    calculate_per_token_loss: bool = False
    barrier_with_L1_time: bool = False
    fp8: Optional[str] = None
    use_dino: bool = False

    def configure_model(self):
        """Instantiate the FoldCompressionModel with this configuration."""
        # Ensure TransformerConfig core fields are aligned
        if getattr(self, 'hidden_size', None) is None:
            self.hidden_size = self.d_model or self.hidden_dim
        if getattr(self, 'num_attention_heads', None) is None:
            self.num_attention_heads = self.n_heads
        if getattr(self, 'num_layers', None) is None:
            self.num_layers = max(self.enc_layers, self.dec_layers)
        if getattr(self, 'sequence_length', None) is None:
            self.sequence_length = self.max_seq_length or 1024
        if getattr(self, 'hidden_dropout', None) is None:
            self.hidden_dropout = self.dropout
        if getattr(self, 'attention_dropout', None) is None:
            self.attention_dropout = self.dropout

        # Pass the full config (self) directly to the model
        return self.model_cls(self)
    
    # def get_loss_reduction_class(self):
    #     """Get the loss reduction class for this configuration.
        
    #     Returns:
    #         Loss reduction class
    #     """
    #     return self.loss_reduction_class

