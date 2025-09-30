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
from src.model.transformer_stack import TransformerStack


class InverseFoldingModel(LanguageModule):
    pre_process: bool = True
    post_process: bool = True
    share_embeddings_and_output_weights: bool = True
    def __init__(self, config: TransformerConfig, 
                 enc_layers, 
                 dec_layers, 
                 hidden_dim,
                 nn_neighbors):
        """ Graph labeling network """
        super(InverseFoldingModel, self).__init__(config)
        self.config: TransformerConfig = config
        self.model_type = ModelType.encoder_or_decoder
        self.struct_encoder = StructureSimEncoder2( enc_layers, hidden_dim, input_node_dim=nn_neighbors*9, scale=10)
        
        
        
        # self.vq_enc = nn.Linear(128, 1280)
        # self.struct_decoder = TransformerStack(
        #     1280, 20, 1, dec_layers, scale_residue=False, n_layers_geom=0, is_geo_attn=False
        # )
        self.seq_decoder = nn.Linear(128, 35)
        
        
        
        
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
                                                      
    def forward(self, position, seq_ids, V, blocks, attn_mask):
        h_V = self.struct_encoder(position, V, blocks, attn_mask)
        # eps = torch.finfo(h_V.dtype).eps
        # h_V = h_V / (torch.norm(h_V, dim=-1, keepdim=True)+eps)
        
        # # predS = self.seq_decoder(h_V)
        # h_V[(seq_ids!=34)]=0
        # h_V = self.struct_decoder(position, self.vq_enc(h_V), attn_mask)
        predS = self.seq_decoder(h_V)
        return predS


FoldCompModelT = TypeVar("FoldCompModelT", bound=InverseFoldingModel)

from dataclasses import dataclass
from typing import Type, Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from src.model.module import StructureDecoder, StructureSimEncoder2





import torch
from typing import Dict, Tuple, Callable, Any
from torch import Tensor
from bionemo.llm.model.loss import _Nemo2CompatibleLossReduceMixin
from megatron.core import parallel_state, tensor_parallel
from nemo.utils import logging
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import (
    MegatronLossReduction,
    masked_token_loss,
    # masked_token_loss_context_parallel,
)


# ============================================================================
# ======== 用户自定义损失函数（请在此处修改/替换实现） ============================
# 函数定义与常规PyTorch损失函数一致，输入一般为logits和labels，输出为【未缩减】tensor或标量
# ============================================================================

def compute_custom_loss(
    output: dict[Tensor],
    batch: dict[Tensor],
    **kwargs: Any,
) -> Tensor:
    """
    用户需要修改：在此实现您的损失计算逻辑。

    Args:
        logits (Tensor): 模型预测的原始输出，形状 [batch, ...]
        labels (Tensor): 真实标签，形状与logits对应
        **kwargs: 其他可能需要的张量，例如权重、mask 等
    Returns:
        Tensor: 未缩减的损失张量，形状 [batch, ...] 或标量
    """
    # return output['predX'].mean()
    # 示例：简单的交叉熵
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(output['predS'].permute(0,2,1), batch['seq_ids'])
    mask = (batch['data_id']!=-1)&(batch['seq_ids']!=34)
    loss = (loss*mask).sum()/mask.sum()  # 返回平均损失
    
    cmp = output['predS'].argmax(dim=-1)==batch['seq_ids']
    recovery = torch.mean((cmp*mask).sum(dim=-1)/mask.sum(-1))
    return loss, {'loss': loss, 'recovery':recovery}
        

    
class CustomLossWithReduction(_Nemo2CompatibleLossReduceMixin, MegatronLossReduction):  # noqa: D101
    def __init__(
        self,
        # =========== 用户根据需要添加或调整参数 ===========
        validation_step: bool = False,
        val_drop_last: bool = True,
        **loss_kwargs: Any,
    ) -> None:
        """初始化自定义Loss模块

        Args:
            validation_step (bool): 是否处于验证阶段
            val_drop_last (bool): 验证时是否丢弃最后一个不满批次
            **loss_kwargs (Any): 传递给 compute_custom_loss 的额外参数
        """
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last
        # 保存到实例供 forward 使用
        self.loss_kwargs = loss_kwargs

    def forward(
        self,
        batch: Dict[str, Tensor],
        forward_out: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        计算并返回带缩减的损失。

        Args:
            batch: 包含输入数据的字典，必须包含 'labels'，可选包含 'loss_mask' 等
            forward_out: 模型前向输出字典，必须包含 'logits'

        Returns:
            Tuple:
                - loss (Tensor): 缩减后的损失，可直接用于反向
                - extras (Dict): 额外信息，如平均损失等
        """

        # if len(forward_out.shape) ==0:
        #     return forward_out, { 'avg': forward_out[None] }
        
        # ======== 用户损失计算入口，不要修改以下调用 ========
        unreduced_loss, _ = compute_custom_loss(
            forward_out,
            batch,
            **self.loss_kwargs
        )  # 张量形状 [batch, ...] 或标量

        # ======== 以下为Nemo框架标准缩减流程 =========
        cp_size = parallel_state.get_context_parallel_world_size()
        # 先按 micro batch 中的 mask 做缩减
        if cp_size == 1:
            loss_mb = masked_token_loss(unreduced_loss, batch.get('loss_mask', None))
        else:
            loss_mb = masked_token_loss_context_parallel(
                unreduced_loss,
                batch.get('loss_mask', None),
                batch.get('num_valid_tokens_in_ub', None),
            )

        # 验证阶段，处理 val_drop_last
        if self.validation_step and not self.val_drop_last:
            num_valid = batch.get('loss_mask', torch.ones_like(unreduced_loss)).sum()
            if loss_mb.isnan():
                if num_valid != 0:
                    raise ValueError("非空输入却得到 NaN 损失")
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
            return loss_mb * cp_size, { 'loss_sum_and_microbatch_size': buf }

        # 正常情况下，跨 data parallel 平均损失并返回额外信息
        reduced = average_losses_across_data_parallel_group([loss_mb])
        return loss_mb * cp_size, { 'avg': reduced }


@dataclass
class InverseFoldingModelConfig(TransformerConfig, iom.IOMixinWithGettersSetters
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
    model_cls: Type[InverseFoldingModel] = InverseFoldingModel
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



    def configure_model(self) -> InverseFoldingModel:
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