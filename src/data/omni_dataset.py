import torch
import msgpack  
import numpy as np
import msgpack
import msgpack_numpy as mn
import random
mn.patch()

from functools import reduce
from typing import Any, Callable, Optional
import numpy as np
import torch
from tqdm import tqdm


class LMDBDataset(torch.utils.data.Dataset):
    coords_eos_array = torch.full(
        (1, 37, 3), torch.inf
    )
    coords_mask_eos_array = torch.full(
        (1, 37, 3), False
    )

    def __init__(
        self, 
        env, 
        cluster_member_idx: dict,
        process_fn: Optional[Callable] = None,
        seq_len: int = 2048,
        seed: int = 42,
        task_type: str = "mlm",
        data_process_fn = None,
        **kwargs
    ):
        self.env = env
        self.process_fn = process_fn
        self.seq_len = seq_len
        self.desc = "LMDBDataset"
        self.rng = random.Random(seed)
        self.task_type = task_type
        self.data_process_fn = data_process_fn
        self.cluster_member_idx = cluster_member_idx

        if not self.env:
            raise IOError(f"Cannot open lmdb dataset")
        # with self.env.begin() as txn:
        #     stat = txn.stat()
        
        keys = []
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for one in tqdm(cursor.iternext(keys=True, values=False)):
                    keys.append(one)

        self.keys = keys
        self.total_samples = len(keys)
        self.cluster_idx_list = list(self.cluster_member_idx.keys())

    def __getitem_based_on_keys__(self, keys):
        unpacked_values = []
        if not isinstance(keys, list):
            keys = [keys]
        with self.env.begin() as txn:
            for key in keys:
                packed_value = txn.get(str(key).encode('utf-8'))
                if packed_value:
                    unpacked_value = msgpack.unpackb(packed_value, raw=False, object_hook=mn.decode)
                    unpacked_value['name']  = key
                    unpacked_values.append(unpacked_value)
        return unpacked_values

    def __len__(self):
        # return self.total_samples
        return int(1e10)
    

    def __getitem__(
        self, 
        idx
    ):
        idx = idx%self.total_samples
        # counter, sample_ids = 0, []
        # rng = np.random.default_rng([idx])
        # while counter < self.seq_len:
        #     rgn = rng.choice(self.cluster_idx_list)
        #     cluster = self.cluster_member_idx.get(str(rgn))
        #     sample_id, sample_length = rng.choice(cluster)
        #     counter += int(sample_length)
        #     sample_ids.append(sample_id)
        
        try:
            if isinstance(idx, int):
                sample_ids = [self.keys[idx].decode()]
            elif isinstance(idx[0], int):
                sample_ids = [self.keys[i].decode() for i in idx]
            values = self.__getitem_based_on_keys__(sample_ids)
            
            return self.data_process_fn(values)
        except:
            return None
    

class LMDBDataset_flatten(torch.utils.data.Dataset):
    coords_eos_array = torch.full(
        (1, 37, 3), torch.inf
    )
    coords_mask_eos_array = torch.full(
        (1, 37, 3), False
    )

    def __init__(
        self, 
        env, 
        process_fn: Optional[Callable] = None,
        total_samples=None,
        seq_len: int = 2048,
        seed: int = 42,
        task_type: str = "mlm",
        data_process_fn = None,
        **kwargs
    ):
        self.env = env
        self.process_fn = process_fn
        self.seq_len = seq_len
        self.desc = "LMDBDataset"
        self.rng = random.Random(seed)
        self.task_type = task_type
        self.data_process_fn = data_process_fn

        if not self.env:
            raise IOError(f"Cannot open lmdb dataset")
        with self.env.begin() as txn:
            stat = txn.stat()
        
        all_num = 0
        keys = []
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for one in tqdm(cursor.iternext(keys=True, values=False)):
                    keys.append(one)
                    all_num += 1
                    if all_num >= total_samples:
                        break

        self.keys = keys
        if total_samples is not None:
            self.total_samples = total_samples
        else:
            self.total_samples = len(keys)

    def __getitem_based_on_keys__(self, keys):
        unpacked_values = []
        if not isinstance(keys, list):
            keys = [keys]
        with self.env.begin() as txn:
            for key in keys:
                packed_value = txn.get(str(key).encode('utf-8'))
                if packed_value:
                    unpacked_value = msgpack.unpackb(packed_value, raw=False, object_hook=mn.decode)
                    unpacked_value = {'name': key, 'value': unpacked_value}
                    unpacked_values.append(unpacked_value)
        return unpacked_values

    def __len__(self):
        # return self.total_samples
        return int(1e10)
    

    def __getitem__(
        self, 
        idx
    ):
        idx = idx%self.total_samples
        
        if isinstance(idx, int):
            sample_ids = [self.keys[idx].decode()]
        elif isinstance(idx[0], int):
            sample_ids = [self.keys[i].decode() for i in idx]
        values = self.__getitem_based_on_keys__(sample_ids)
        
        return values[0]
    

def split_ds(
        cluster_msg_file: str,
        split='949, 50, 1', 
        seed=1130):
    number_strings = split.split(',')

    split = [int(num) for num in number_strings]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split, dtype=np.float32)
    split /= split.sum()
    # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ds: {len(ds)} {block_size}")

    with open(cluster_msg_file, "rb") as f:
        cluster_idx = msgpack.unpack(f, raw=False)
    clusters = list(cluster_idx.keys())
    n_clusters = len(cluster_idx)
    # clusters = np.random.permutation(clusters)
    clusters = sorted(clusters)
    train_cluster_ids = clusters[:int(n_clusters*split[0])]
    val_cluster_ids = clusters[int(n_clusters*split[0]):int(n_clusters*split[:2].sum())]
    test_cluster_ids = clusters[int(n_clusters*split[:2].sum(0)):]
    train_clusters = { k: cluster_idx.get(k) for k in train_cluster_ids}
    val_clusters = { k: cluster_idx.get(k) for k in val_cluster_ids}
    test_clusters = { k: cluster_idx.get(k) for k in test_cluster_ids}

    return train_clusters, val_clusters, test_clusters


from torch import Tensor

def batched_topk_neighbors_3d(
    X: Tensor,            # [B, L, 3], float
    mask: Tensor,         # [B, L],   bool  True=valid, False=padding
    K: int
) -> Tensor:
    """
    返回邻居索引 indices: [B, L, K]
    规则：
      - 只在同一 batch 内寻找邻居
      - 仅考虑 mask=True 的位置为候选邻居
      - 自身不计为邻居；若候选不足 K，用自身索引填充
      - 距离相等时按索引升序打破平手
      - 对于 mask=False 的查询位置，结果全为自身索引
    """
    assert X.dim() == 3 and X.size(-1) == 3, "X should be [B, L, 3]"
    assert mask.dim() == 2 and mask.shape[:2] == X.shape[:2], "mask should be [B, L]"
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    # --- 计算两两平方距离：d(i,j) = ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    x2 = (X * X).sum(-1)                    # [B, L]
    # batch 内点乘
    G = torch.bmm(X, X.transpose(1, 2))     # [B, L, L]
    dist2 = x2.unsqueeze(2) + x2.unsqueeze(1) - 2.0 * G  # [B, L, L]
    # 数值稳定：避免出现极小负数
    dist2 = dist2.clamp_min_(0)

    # --- 掩码处理：无效列(候选邻居 j)设为 +inf；无效行(查询 i)整行设为 +inf；对角线设为 +inf（排除自身）
    INF = torch.tensor(float("inf"), device=device, dtype=dtype)
    # 无效列（j 不可用）
    col_mask = mask.unsqueeze(1).expand(B, L, L)         # broadcast 到列
    dist2 = dist2.masked_fill(~col_mask, INF)

    # 无效行（i 不可用）
    row_mask = mask.unsqueeze(2).expand(B, L, L)         # broadcast 到行
    dist2 = dist2.masked_fill(~row_mask, INF)

    # 排除自身：对角线设为 +inf（仅对有效行有意义，无效行已是 +inf）
    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)  # [1, L, L]
    dist2 = dist2.masked_fill(eye, INF)

    # --- 距离相等时按索引升序打破平手：给每个“列 j”加极小权重 eps * j
    # 这样更小的 j 拥有更小的加权，保持“按距离升序 + 索引升序”的排序
    # 注意：对 inf 加任意有限数仍为 inf，不影响掩码逻辑
    idxs = torch.arange(L, device=device, dtype=dtype).view(1, 1, L)  # 作为列的 j
    # 选择很小的 eps，远小于距离量级；用 1e-7 通常足够，避免改变真实排序
    eps = torch.tensor(1e-7, device=device, dtype=dtype)
    dist2 = dist2 + eps * idxs  # 广播到 [B, L, L]

    # --- 取每个位置的最小 K 个（由近到远）
    # torch.topk 支持 largest=False 直接取最小值
    # 即使有 inf，也会被排到后面
    topk_vals, topk_idx = torch.topk(dist2, k=min(K, L), dim=-1, largest=False)  # [B, L, K'], K'<=L

    # 若 K > L-1（极端情况），topk 仍然会给出最多 L 个下标；我们统一裁成 K 列
    if topk_idx.size(-1) != K:
        # pad 到 K 列，用占位后再填自索引
        pad_cols = K - topk_idx.size(-1)
        pad = torch.full((B, L, pad_cols), 0, device=device, dtype=torch.long)
        topk_idx = torch.cat([topk_idx, pad], dim=-1)
        topk_vals = torch.cat([topk_vals, torch.full((B, L, pad_cols), INF, device=device, dtype=dtype)], dim=-1)

    # --- 将不可用的位置（值为 inf）替换为“自身索引”
    self_idx = torch.arange(L, device=device).view(1, L, 1).expand(B, L, K)  # [B, L, K]
    is_finite = torch.isfinite(topk_vals)  # [B, L, K]
    # 对于查询行本身无效（mask=False）的情况，整行 topk_vals 都是 inf -> 全部替换为自身
    indices = torch.where(is_finite, topk_idx, self_idx)

    return indices  # [B, L, K]

from torch import Tensor
def index_along_len_tad(X: Tensor, idx: Tensor) -> Tensor:
    B, L, H, C = X.shape
    X_expanded = X.unsqueeze(2).expand(B, L, idx.size(2), H, C)             # [B,L,K,H,3]
    idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand_as(X_expanded)    # [B,L,K,H,3]
    return torch.take_along_dim(X_expanded, idx_expanded, dim=1)            # [B,L,K,H,3]