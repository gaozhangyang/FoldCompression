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
    masked_token_loss_context_parallel,
)
from src.model.chroma.struct_loss import ReconstructionLosses
from src.model.chroma.transforms import transform_cbach_to_sbatch

# ============================================================================
# ======== 用户自定义损失函数（请在此处修改/替换实现） ============================
# 函数定义与常规PyTorch损失函数一致，输入一般为logits和labels，输出为【未缩减】tensor或标量
# ============================================================================
def compute_custom_loss(
    output: dict[Tensor],
    batch: dict[Tensor],
    mode='train',
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
    if mode == 'train':
        pred_X = output['predX']
        V = output['v']
        loss = torch.sum((pred_X - V)**2, dim=-1).mean()
        return loss, {'loss': loss}  # 返回未缩减的损失张量
    else:
        pred_X = output['predX']
        V = batch['value']
        loss = torch.sum((pred_X - V)**2, dim=-1).mean()       
        out = {'loss': loss}
        return out['loss'], out  # 返回未缩减的损失张量
    
        


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
        mode='train'
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

        # ======== 用户损失计算入口，不要修改以下调用 ========
        unreduced_loss, _ = compute_custom_loss(
            forward_out,
            batch,
            mode,
            **self.loss_kwargs
        )  # 张量形状 [batch, ...] 或标量

        # ======== 以下为Nemo框架标准缩减流程 =========
        cp_size = parallel_state.get_context_parallel_world_size()
        # 先按 micro batch 中的 mask 做缩减
        if cp_size == 1:
            loss_mb = unreduced_loss
        else:
            loss_mb = unreduced_loss
            torch.distributed.all_reduce(loss_mb, group=parallel_state.get_context_parallel_group())

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
