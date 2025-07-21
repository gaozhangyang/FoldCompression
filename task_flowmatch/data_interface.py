import os
from typing import Literal
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from bionemo.esm2.data import dataset, tokenizer
from src.data.omni_dataset import LMDBDataset, LMDBDataset_flatten, split_ds
import lmdb
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from src.interface.data_interface_base import DataInterfaceBase
from typing import Dict, Any


Mode = Literal["train", "validation", "test"]



class ESMDataModule(DataInterfaceBase):
    """LightningDataModule wrapper of `ESMDataset`."""

    def __init__(
        self,
        cluster_path: str | os.PathLike,
        database_path: str | os.PathLike,
        seed: int | None = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,  # TODO(@jomitchell) can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        random_mask_strategy: dataset.RandomMaskStrategy = dataset.RandomMaskStrategy.ALL_TOKENS,
        tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
        dataloader_type: Literal["single", "cyclic"] = "single",
        noise_scale = 0.1,
        prefix_len = 6,
        data_splits: str = '9990, 5, 5',  # train, val, test
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # del self.hparams['optimizer'] # optimizer do not support serialization
        self._seed = seed
        self._min_seq_length = min_seq_length if min_seq_length is not None else max_seq_length
        self._max_seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._mask_random_prob = mask_random_prob
        self._random_mask_strategy = random_mask_strategy
        self._tokenizer = tokenizer

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory
        self.seq_len = max_seq_length
        self.setup()
        
        

    def setup(self, stage: str = "") -> None:
        
        self.env = lmdb.open(self.hparams.database_path, 
                        readonly=True, 
                        lock=False, 
                        readahead=True, 
                        meminit=True, 
                        create=False, 
                        map_size=10**10)
        
        train_cluster, val_cluster, test_cluster = split_ds(self.hparams.cluster_path, self.hparams.data_splits, seed=self.hparams.seed)
        
        self.train_cluster = train_cluster
        self.val_cluster = val_cluster
        self.test_cluster = test_cluster

        self.data_sampler = MegatronDataSampler(
            seq_len=self.hparams.max_seq_length,
            micro_batch_size=self.hparams.micro_batch_size,
            global_batch_size=self.hparams.global_batch_size,
            dataloader_type=self.hparams.dataloader_type,
            rampup_batch_size=self.hparams.rampup_batch_size,
        )
        

    def train_dataloader(self):
        return self.build_pretraining_data_loader(self.train_cluster, mode='train')

    def val_dataloader(self):
        return self.build_pretraining_data_loader(self.test_cluster, mode='val')

    def test_dataloader(self):
        return self.build_pretraining_data_loader(self.test_cluster, mode='test')

    def build_pretraining_data_loader(self, cluster_idx, mode='train'):
        # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        # self.batch_sampler = ClusterLayeredSampler(
        #         sampler=sampler,
        #         micro_batch_size=self.hparams.micro_batch_size,
        #         cluster_member_idx=cluster_idx,
        #         seed=self.hparams.seed,
        #         mode=mode
        #     )
        
        
        if mode == 'train':
            # total_samples = 30000000
            total_samples = 100000
        else:
            total_samples = 1000
        dataset = LMDBDataset_flatten(self.env, 
                              total_samples=total_samples, seq_len=self.hparams.max_seq_length, 
                              process_fn=None, seed=self.hparams.seed, task_type='mlm')
        loader = DataLoader(dataset,
                            num_workers=self.hparams.num_workers, # 
                            pin_memory=True,
                            collate_fn=self.data_process_fn)
        return loader
    
    def data_process_fn(self, data_list):
        data_list = [one for one in data_list if one is not None]
       
        values = torch.stack([torch.tensor(data_list[idx]['value']) for idx in range(len(data_list))], dim=0)
        names = [data_list[idx]['name'] for idx in range(len(data_list))]
        
        return {
            'value': values,
            'name': names,
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Dump your sampler’s state into the trainer checkpoint
        if self.batch_sampler is not None:
            checkpoint["train_sampler_state"] = self.batch_sampler.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # When restoring, reload that state back into your sampler
        sampler_state = checkpoint.get("train_sampler_state", None)
        if sampler_state is not None and self.batch_sampler is not None:
            self.batch_sampler.load_state_dict(sampler_state)
    
def Segment(L, n, k):
    """
    将长度为L的序列划分为n个连续片段，每个片段长度至少为10，并从这些片段中随机选择k个片段。

    参数:
    L (int): 序列的总长度。
    n (int): 片段的数量。
    k (int): 需要选择的片段数量。

    返回:
    List[List[int]]: 被选中片段的索引列表。每个片段是一个整数列表，表示片段中的所有索引位置。

    异常:
    ValueError: 如果L不足以划分n个长度至少为5的片段，或选择的片段数量k超过总片段数n，将引发异常。
    """
    # 检查序列长度是否足够
    if n * 10 > L:
        unmask = list(range(L))
        return unmask
    
    start = 0
    segments = []
    
    # 划分n个片段
    for i in range(n):
        if i == n - 1:
            # 最后一个片段包含剩余的所有元素
            segments.append(range(start, L))
        else:
            # 随机确定每个片段的结束位置
            end = random.randint(start + 10, L - (n - i - 1) * 10)
            segments.append(range(start, end))
            start = end
    
    # 从n个片段中随机选择k个片段，保证选择的片段互不重复
    if k > n:
        raise ValueError("选择的片段数量k不能超过总片段数n")
    
    selected_segments = random.sample(segments, k)
    
    unmask = [list(seg) for seg in selected_segments]
    unmask = [item for sublist in unmask for item in sublist]
    
    # 输出每个被选中片段的索引列表
    return sorted(unmask)#, mask
