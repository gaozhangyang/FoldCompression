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
from src.model.chroma.struct_loss import ReconstructionLosses
from src.model.chroma.transforms import transform_cbach_to_sbatch

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
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # return loss_fn(logits, labels)
    pred_X, chain, X_true, prefix_len = output['predX'], batch['data_id'], batch['coords'][:,:,:5],  batch['prefix_len']
    pred_X0 = output.get('predX0', None)
    
    struct_loss = ReconstructionLosses(
            rmsd_method='symeig', loss_scale=10.0
        )
    
    C_batch = chain
    X_true_batch = X_true
    
    out = compute_loss(X_true_batch, pred_X, struct_loss, C_batch, prefix_len)
    # out0 = compute_loss(X_true_batch, pred_X0, struct_loss, C_batch, 0)
    
    loss = out['loss'] #+ out0['loss']
    return loss, out  # 返回未缩减的损失张量
        

def compute_loss(X_true_batch, pred_X_batch, struct_loss, C_batch, prefix_len):
    B,L = C_batch.shape
    X_true_batch = X_true_batch.reshape(B,L,-1,3)
    pred_X_batch = pred_X_batch.reshape(B,L,-1,3)
    mask_batch = torch.isnan(X_true_batch.sum(dim=(-2,-1)))
    C_batch[mask_batch]=-1
    pred_X_batch[mask_batch]=0
    X_true_batch[mask_batch]=0

    results = struct_loss(pred_X_batch[:,prefix_len:], X_true_batch[:,prefix_len:], C_batch[:,prefix_len:])
                
    
    out = {}
    loss = 0
    for key in ['batch_global_mse', 'batch_fragment_mse', 'batch_pair_mse', 'batch_neighborhood_mse', 'batch_distance_mse']:
        if results.get(key):
            loss += results[key]
            out.update({key: results[key]})
    out['loss'] = loss
    return out
    

