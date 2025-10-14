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
    # masked_token_loss_context_parallel,
)
from nemo.utils import logging

# BioNeMo imports
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from bionemo.llm.utils import iomixin_utils as iom

# Local imports
from src.model.chroma.struct_loss import ReconstructionLosses
from src.model.chroma.transforms import transform_cbach_to_sbatch
from src.model.module import StructureDecoder, StructureSimEncoder2


class FoldCompressionModel(LanguageModule):
    """Graph labeling network for protein structure compression."""
    
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    
    def __init__(self, config: TransformerConfig, enc_layers: int, dec_layers: int, 
                 hidden_dim: int, nn_neighbors: int):
        """Initialize the FoldCompressionModel.
        
        Args:
            config: Transformer configuration
            enc_layers: Number of encoder layers
            dec_layers: Number of decoder layers
            hidden_dim: Hidden dimension size
            nn_neighbors: Number of nearest neighbors
        """
        super(FoldCompressionModel, self).__init__(config)
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        
        self.struct_encoder = StructureSimEncoder2(
            enc_layers, hidden_dim, input_node_dim=nn_neighbors*9, scale=100
        )
        self.struct_decoder = StructureDecoder(n_layers=dec_layers)
    
    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func.
        """
        self.input_tensor = input_tensor
    
    def forward(self, position, seq_ids, V, blocks, attn_mask, infer_feats=False):
        """Forward pass of the model.
        
        Args:
            position: Position embeddings
            seq_ids: Sequence IDs
            V: Input features
            blocks: Block structure
            attn_mask: Attention mask
            infer_feats: Whether to only return features
            
        Returns:
            Predicted coordinates or features
        """
        h_V = self.struct_encoder(position, V, blocks, attn_mask)
        eps = torch.finfo(h_V.dtype).eps
        h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True) + eps)
        
        if infer_feats:
            return h_V

        h_V[seq_ids != 34] = 0
        predX, h_V = self.struct_decoder(position, h_V, attn_mask)
        
        return predX
    
    @classmethod
    def compute_custom_loss(
        cls,
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
        pred_X = output['predX']
        chain = batch['data_id']
        X_true = batch['coords'][:, :, :5]
        prefix_len = batch['prefix_len']
        
        struct_loss = ReconstructionLosses(rmsd_method='symeig', loss_scale=10.0)
        
        out = cls.compute_loss(X_true, pred_X, struct_loss, chain, prefix_len)
        
        return out['loss'], out
        
    @classmethod
    def compute_loss(cls, X_true_batch: Tensor, pred_X_batch: Tensor, 
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
        out = {}
        loss = 0
        loss_keys = ['batch_global_mse', 'batch_fragment_mse', 'batch_pair_mse', 
                    'batch_neighborhood_mse', 'batch_distance_mse']
        
        for key in loss_keys:
            if results.get(key):
                loss += results[key]
                out[key] = results[key]
        
        out['loss'] = loss
        return out


class CustomLossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):
    """Custom loss reduction class for FoldCompression model."""
    
    def __init__(
        self,
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
        unreduced_loss, _ = FoldCompressionModel.compute_custom_loss(
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
FoldCompModelT = TypeVar("FoldCompModelT", bound=FoldCompressionModel)


@dataclass
class FoldCompressionConfig(TransformerConfig, iom.IOMixinWithGettersSetters):
    """Configuration for FoldCompressionModel.

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
        """Instantiate the FoldCompressionModel with this configuration.
        
        Returns:
            Configured FoldCompressionModel instance
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
            hidden_dim=self.hidden_dim,
            nn_neighbors=9  # Default value, should be configurable
        )
        return model
    
    def get_loss_reduction_class(self) -> Type[MegatronLossType]:
        """Get the loss reduction class for this configuration.
        
        Returns:
            Loss reduction class
        """
        return self.loss_reduction_class

