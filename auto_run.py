import torch
import torch.multiprocessing as mp
import os
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
def occupy_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Initializing...")

    # 你可以手动调这个大小撑满显存
    size = 60000  # 大概 4GB*3 的矩阵：float32，占显存超快

    # 创建占显存的大张量
    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    print(f"[GPU {gpu_id}] Start computing...")

    while True:
        # 无限爆算循环
        c = torch.matmul(a, b)
        c = torch.sin(c)
        a = c + 0.0001 * a
        torch.cuda.synchronize(device)  # 强制同步，GPU不能偷懒

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")

    
    mp.set_start_method("spawn")

    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(target=occupy_gpu, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
