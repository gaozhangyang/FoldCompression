import lmdb
import pickle
import torch
import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()  # 启用对 numpy 的支持

if __name__ == "__main__":
    lmdb_path = "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/StructCompression/results/struct_compress/baseline_prefix32_len512_dec1_1M_bs32_run3_infer/compression_data.lmdb"

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            name = key.decode('utf-8')  # LMDB key 是 bytes
            # tensor = pickle.loads(value)  # 反序列化为 PyTorch tensor
            array = msgpack.unpackb(value, raw=False, object_hook=m.decode)
            array = torch.from_numpy(array)

            # 示例输出：name 和 tensor shape
            print(f"{name}: {array.shape}")
    
