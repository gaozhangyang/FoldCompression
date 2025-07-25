import os
from typing import Literal
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from bionemo.esm2.data import dataset, tokenizer
from src.data.omni_dataset import LMDBDataset, split_ds
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
        # env = lmdb.open('/nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem.db', 
        #                 readonly=True, 
        #                 lock=False, 
        #                 readahead=True, 
        #                 meminit=True, 
        #                 create=False, 
        #                 map_size=1099511627776)
        
        # train_cluster, val_cluster, test_cluster = split_ds('/nfs_beijing_os/linlinchao/afdb/rep_mem_v2/afdb_rep_mem-cluster.msgpack', '9990, 5, 5', seed=self.hparams.seed)
        
        self.env = lmdb.open(self.hparams.database_path, 
                        readonly=True, 
                        lock=False, 
                        readahead=True, 
                        meminit=True, 
                        create=False, 
                        map_size=1099511627776)
        
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
        
        self.update_init_global_step()
        
        dataset = LMDBDataset(self.env, 
                              self.train_cluster, seq_len=self.hparams.max_seq_length, 
                              process_fn=None, seed=self.hparams.seed, task_type='mlm', data_process_fn=lambda x: self.data_process_fn(x, mode=mode))
        loader = DataLoader(dataset,
                            num_workers=self.hparams.num_workers, # 
                            pin_memory=True)
        return loader
    
    def data_process_fn(self, data_list, mode='train'):
        segment_num = 5
        noise_scale = self.hparams.noise_scale
        random.shuffle(data_list)
        data_id = []
        seq_ids, struct_ids, coords, lens = [], [], [], []
        position = []
        blocks_list, types_list = [], []
        names = []
        start = 0
        dataidx = 0
        for data in data_list:
            L = data['seq_ids'].shape[0]
            if mode=='train':
                n = int(torch.randint(1, segment_num+1, (1,)))
                k = int(torch.randint(1, n+1, (1,)))
            else:
                n=1
                k=1
            unmasked = torch.tensor(Segment(L, n, k))
            L = unmasked.shape[0] + self.hparams.prefix_len
            
            if (start+L > self.seq_len) and (start<self.seq_len):
                L = self.seq_len-start
                seq_id_tmp = torch.from_numpy(data['seq_ids'])[unmasked][:L-self.hparams.prefix_len]
                coords_tmp = torch.from_numpy(data['cords'])[0,unmasked][:L-self.hparams.prefix_len]
            else:
                seq_id_tmp = torch.from_numpy(data['seq_ids'])[unmasked]
                coords_tmp = torch.from_numpy(data['cords'])[0,unmasked]
            
            # coords_tmp[torch.isnan(coords_tmp)] = 0
            coords_tmp = torch.nan_to_num(coords_tmp, nan=0.0)
            # add k prefix tokens
            seq_id_tmp = torch.cat([torch.zeros((self.hparams.prefix_len,), dtype=torch.int64)+34, seq_id_tmp], dim=0)
            coords_tmp = torch.cat([torch.zeros((self.hparams.prefix_len, 37, 3)), coords_tmp], dim=0)
                
            position.append(torch.cat([-torch.arange(1, 1+self.hparams.prefix_len), unmasked]))
            blocks = coords_tmp[..., :4, :]+noise_scale*torch.randn_like(coords_tmp[..., :4, :])
            types = torch.arange(blocks.shape[1])[None].repeat(blocks.shape[0],1)
            blocks_list.append(blocks)
            types_list.append(types)
            seq_ids.append(seq_id_tmp)
            coords.append(coords_tmp)
            data_id.append(torch.ones_like(seq_id_tmp)*dataidx+1)
            names.append(data['name'])
            dataidx += 1
            
            lens.append(L)
            start += L
            if start >= self.seq_len:
                break
            
        
        seq_ids = torch.cat(seq_ids)
        coords = torch.cat(coords, dim=0)
        
        assert coords.shape[0] == seq_ids.shape[0]
        blocks = torch.cat(blocks_list, dim=0)
        # blocks[torch.isnan(blocks)] = 0
        types = torch.cat(types_list, dim=0)
        data_id = torch.cat(data_id)
        position = torch.cat(position, dim=0)+self.hparams.prefix_len
        loss_mask = position>=6
        
        
        if seq_ids.shape[0]<self.seq_len:
            blocks = F.pad(blocks, (0,0,0,0,0, self.seq_len-blocks.shape[0]))
            types = F.pad(types, (0,0,0, self.seq_len-types.shape[0]))
            seq_ids = F.pad(seq_ids, (0, self.seq_len-seq_ids.shape[0]))
            coords = F.pad(coords, (0,0,0,0,0,self.seq_len-coords.shape[0]))
            lens.append(self.seq_len-seq_ids.shape[0])
            loss_mask = F.pad(loss_mask, (0, self.seq_len-loss_mask.shape[0]))
            data_id = F.pad(data_id, (0, self.seq_len-data_id.shape[0]), value=-1)
            position = F.pad(position, (0, self.seq_len-position.shape[0]), value=-1)
        else:
            blocks = blocks[:self.seq_len]
            types = types[:self.seq_len]
            seq_ids = seq_ids[:self.seq_len]
            coords = coords[:self.seq_len]
            loss_mask = loss_mask[:self.seq_len]
            data_id = data_id[:self.seq_len]
            position = position[:self.seq_len]
            
        lens = torch.tensor(lens)
        lens = F.pad(lens, (0, self.seq_len-lens.shape[0]))
        
        
        return {'names': names,
                'seq_ids': seq_ids,
                'coords': coords,
                'division': lens,
                'loss_mask': loss_mask,
                'blocks': blocks,
                'data_id': data_id,
                'position': position,}

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
