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
        
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                keys = list(cursor.iternext(keys=True, values=False))

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
        return int(1e8)
    

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
        
        if isinstance(idx, int):
            sample_ids = [self.keys[idx].decode()]
        elif isinstance(idx[0], int):
            sample_ids = [self.keys[i].decode() for i in idx]
        values = self.__getitem_based_on_keys__(sample_ids)
        
        return self.data_process_fn(values)
    



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

