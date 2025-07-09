import torch
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence, TypedDict, cast
from torch import Tensor
from megatron.core.packed_seq_params import PackedSeqParams

# 顶层函数，multiprocessing 才能 pickle
def _worker(args):
    fn, d, kwargs = args
    return fn(*d, **kwargs)  # d 必须是 tuple；如果是单参数就传成 (d,) 即可


from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    if n_jobs is None:
        n_jobs = cpu_count()

    # 定义一个真正可以 pickling 的函数，避免 lambda 引起问题
    def _wrapped(d):
        return pickleable_fn(*d, **kwargs)

    # tqdm 外部包裹，不要嵌入 generator 里
    data_iter = list(tqdm(data, desc=desc))

    with parallel_backend('loky'):  # 或 'multiprocessing'
        results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
            delayed(_wrapped)(d) for d in data_iter
        )
    return results


def flatten_dict(d, parent_key='', sep='.', level=0):
    """
    递归地将嵌套字典拉平为一个单层字典，取消第一级父键。

    :param d: 输入的嵌套字典
    :param parent_key: 父键（用于递归）
    :param sep: 键之间的分隔符，默认为点号 '.'
    :param level: 当前递归的层级（用于取消第一级父键）
    :return: 拉平后的单层字典
    """
    items = {}
    for k, v in d.items():
        # 构建新的键
        if level <=1:
            new_key = k  # 第一级取消父键
        else:
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # 如果值是字典，递归拉平
            items.update(flatten_dict(v, new_key, sep=sep, level=level + 1))
        else:
            # 否则直接添加到结果中
            items[new_key] = v
    return items

def process_args(parser, config_path):
    from hydra import initialize, compose
    from omegaconf import DictConfig, OmegaConf
    import sys
    def eval_resolver(expr: str):
        return eval(expr, {}, {})

    OmegaConf.register_new_resolver("eval", eval_resolver, use_cache=False)

    # ----------------------------------------------------------------------------
    # 1) 先不看命令行，拿到纯“parser 默认值”：
    defaults = parser.parse_args([])
    defaults_dict = vars(defaults)

    # 2) 真正去解析一次命令行（CLI + 默认）：
    args = parser.parse_args()
    args_dict = vars(args)

    # 3) 再读你的 Hydra config，平展开成普通 dict：
    with initialize(config_path=config_path):
        cfg: DictConfig = compose(config_name=args.config_name)
    config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # ----------------------------------------------------------------------------
    # 4) 挖出哪些 key 的值是“真由用户在命令行里指定”的：
    passed = set()
    for tok in sys.argv[1:]:
        if not tok.startswith('--'):
            continue
        # 支持 --foo=bar 和 --foo bar 两种写法
        key = tok.lstrip('-').split('=')[0].replace('-', '_')
        passed.add(key)

    # 5) 最终合并：CLI > config_file > parser_default
    merged = {}
    for key in set(list(defaults_dict.keys())+list(config_dict.keys())):
        if key in passed:
            # 用户显式传进来的
            merged[key] = args_dict[key]
        elif key in config_dict:
            # config 文件里有，且用户没在 CLI 指定，就用它
            merged[key] = config_dict[key]
        else:
            # 都没有指定，就退回 parser 默认
            merged[key] = defaults_dict[key]

    # 用合并后的结果更新 args Namespace
    args.__dict__.update(merged)
    return args


class SequenceBatchCore(TypedDict):
    """Input datatype for inference with BERT-like models."""

    cu_seqlens: Tensor


class SequenceBatch(SequenceBatchCore, total=False):
    """Input datatype for inference with BERT-like models."""

    cu_seqlens_argmin: Tensor
    max_seqlen: Tensor
    
def get_packed_seq_params(batch: SequenceBatch) -> PackedSeqParams:
    """Get the packed sequence parameters for the given batch.

    This function should only be called if `cu_seqlens` is defined in the batch.

    Args:
        batch: The input batch to pack.

    Returns:
        PackedSeqParams: The packed sequence parameters containing the following attributes:
            - cu_seqlens_q (Tensor): The sequence lengths for query.
            - cu_seqlens_kv (Tensor): The sequence lengths for key and value.
            - max_seqlen_q (Tensor, optional): The maximum sequence length for query.
            - max_seqlen_kv (Tensor, optional): The maximum sequence length for key and value.
            - qkv_format (str): The format of query, key, and value tensors.

    """
    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )