# burn_all.py
# -*- coding: utf-8 -*-
import os
import math
import time
import signal
import argparse
import multiprocessing as mp

import torch

# ---------------------- util ----------------------
def bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def align_to(x: int, a: int) -> int:
    return max(a, (x // a) * a)

def pick_square_dim_from_mem_safe(target_bytes, dtype, matrices=3, align=128, safety=0.90):
    """
    给定目标显存预算（字节），估算 d，使 matrices * d*d * itemsize ≈ target_bytes*safety
    """
    item = bytes_per_elem(dtype)
    target_bytes = int(target_bytes * safety)
    d = int(math.sqrt(max(1, target_bytes / (matrices * item))))
    return align_to(d, align)

def touch_tensor(t):
    # 触碰数据，确保物理驻留，避免按需分配导致的突发 OOM
    if t.numel() == 0:
        return
    step = max(1, t.numel() // 2048)
    _ = t.view(-1)[::step].sum().item()

def cpu_burn_worker(cpu_dim, dtype, stop_flag: mp.Event):
    # 纯 CPU GEMM + 非线性，走 MKL/OPENBLAS 多线程
    a = torch.randn((cpu_dim, cpu_dim), dtype=dtype, device="cpu")
    b = torch.randn((cpu_dim, cpu_dim), dtype=dtype, device="cpu")
    while not stop_flag.is_set():
        c = a @ b
        c = torch.sin(c)
        a = c + 1e-4 * a

def allocate_gemm_tensors_with_retry(device, dtype, d_init, matrices=3, max_trials=10, shrink=0.90, align=128):
    """
    一次性大块 flat 分配，减少碎片；失败则退让缩小 d 重试。
    返回: a, b, reserve, d_real
    """
    g = torch.Generator(device=device).manual_seed(1234)
    d = d_init
    last_err = None
    for t in range(max_trials):
        try:
            elems = matrices * d * d
            flat = torch.empty(elems, device=device, dtype=dtype)
            flat.normal_(0.0, 1.0, generator=g)
            stride = d * d
            a = flat[:stride].view(d, d)
            b = flat[stride:2 * stride].view(d, d)
            reserve = flat[2 * stride:]  # 作为占位/碎片填充
            torch.cuda.synchronize(device)
            return a, b, reserve, d
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            try:
                del flat
            except Exception:
                pass
            torch.cuda.empty_cache()
            d = align_to(int(d * shrink), align)
            time.sleep(0.05)
    raise RuntimeError(f"Failed to allocate GEMM tensors after {max_trials} tries (last dim={d}).") from last_err

# ---------------------- main worker ----------------------
def occupy_gpu(
    gpu_id: int,
    mem_frac: float = 0.80,
    dtype_str: str = "float32",
    host_mem_gb: float = 8.0,
    cpu_dim_hint: int = 8192,
    h2d_mb_per_iter: int = 512,
    use_fp16_gemm: bool = False,
    stop_event: mp.Event = None,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # 限制每进程可用显存比例，进一步缓解 OOM
    try:
        torch.cuda.set_per_process_memory_fraction(min(0.95, mem_frac + 0.05), device=gpu_id)
    except Exception:
        pass

    # 选择 dtype
    if use_fp16_gemm:
        dtype = torch.float16
    else:
        dtype = getattr(torch, dtype_str)

    print(f"[GPU {gpu_id}] Init on {device}, dtype={dtype} …")

    # CPU 线程配额：按 GPU 数平均
    total_cpus = os.cpu_count() or 8
    num_gpus = max(1, torch.cuda.device_count())
    per_gpu_threads = max(1, total_cpus // num_gpus)
    os.environ.setdefault("OMP_NUM_THREADS", str(per_gpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(per_gpu_threads))
    torch.set_num_threads(per_gpu_threads)
    torch.set_num_interop_threads(max(1, per_gpu_threads // 2))
    print(f"[GPU {gpu_id}] CPU threads: {torch.get_num_threads()}  OMP/MKL={os.environ['OMP_NUM_THREADS']}")

    # 估算 d
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    target_bytes = int(min(free_bytes, total_bytes) * mem_frac)
    d = pick_square_dim_from_mem_safe(target_bytes, dtype=dtype, matrices=3, align=128, safety=0.90)
    print(f"[GPU {gpu_id}] total={total_bytes/1e9:.1f}GB free={free_bytes/1e9:.1f}GB -> target≈{target_bytes/1e9:.1f}GB, try dim={d}")

    # 分配（有退让重试）
    a, b, reserve, d = allocate_gemm_tensors_with_retry(device, dtype, d_init=d, matrices=3, max_trials=10)
    print(f"[GPU {gpu_id}] Allocated: dim={d} (A,B) + reserve({reserve.numel()*bytes_per_elem(dtype)/1e9:.2f}GB)")

    # 分配并触碰大量 Host 内存（非 pinned + pinned）
    host_bytes = int(host_mem_gb * (1024**3))
    host_f32 = torch.empty(host_bytes // 4, dtype=torch.float32)  # non-pinned
    touch_tensor(host_f32)
    pinned_cap_gb = min(host_mem_gb, 4.0)  # pinned 上限 4GB，太大可能失败
    pinned_bytes = int(pinned_cap_gb * (1024**3))
    pinned = torch.empty(pinned_bytes // 4, dtype=torch.float32, pin_memory=True)
    touch_tensor(pinned)
    print(f"[GPU {gpu_id}] Host RAM reserved: {host_mem_gb:.1f}GB (non-pinned) + {pinned_bytes/1e9:.1f}GB (pinned)")

    # CPU burn 子进程
    cpu_dtype = torch.float32
    cpu_dim = align_to(max(2048, cpu_dim_hint), 128)
    cpu_stop = mp.Event()
    cpu_proc = mp.Process(target=cpu_burn_worker, args=(cpu_dim, cpu_dtype, cpu_stop))
    cpu_proc.start()
    print(f"[GPU {gpu_id}] CPU GEMM started: dim={cpu_dim}")

    # H2D/D2H 压力
    stream = torch.cuda.Stream(device=device)
    h2d_elems = (h2d_mb_per_iter * 1024 * 1024) // 4  # float32 元素数
    h2d_buf = torch.randn(h2d_elems, dtype=torch.float32, pin_memory=True)
    d2h_buf = torch.empty_like(h2d_buf)

    # 主循环：若某次计算 OOM，自动把 d 缩小 10% 并重建
    shrink_factor = 0.90
    grow_factor = 1.03  # 偶尔温和尝试增大 reserve，以吃掉空闲（失败会再缩回）
    iter_cnt = 0

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            try:
                # 1) GPU GEMM + 非线性
                c = a @ b
                c = torch.sin(c)
                a = c + (5e-4 if dtype == torch.float16 else 1e-4) * a

                # 2) 异步 H2D/D2H
                with torch.cuda.stream(stream):
                    d_tmp = h2d_buf.to(device, non_blocking=True)
                    _ = (d_tmp * 1.0001).sum()
                    d2h_buf.copy_(d_tmp.to("cpu", non_blocking=True), non_blocking=True)

                # 3) 触碰 host 内存（维持占用）
                host_f32[::4096] = host_f32[::4096] * 1.000001 + 1e-6
                pinned[::4096] = pinned[::4096] * 0.999999 + 1e-6

                torch.cuda.synchronize(device)

                # 4) 偶尔尝试扩大 reserve（温和增压）
                iter_cnt += 1
                if iter_cnt % 200 == 0:
                    try:
                        extra = int(reserve.numel() * (grow_factor - 1.0))
                        if extra > 0:
                            # 试图扩大 flat 占位：直接再分一块小 buffer
                            extra_buf = torch.empty(extra, device=device, dtype=dtype)
                            # 与 reserve 合并管理：这里简单保留引用，避免被 GC
                            reserve = torch.cat([reserve, extra_buf])
                    except torch.cuda.OutOfMemoryError:
                        # 增压失败就忽略
                        torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                # 本轮计算 OOM：降 d 并重建
                print(f"[GPU {gpu_id}] OOM in loop, shrinking …")
                try:
                    del a, b, reserve
                except Exception:
                    pass
                torch.cuda.empty_cache()
                d = align_to(int(d * shrink_factor), 128)
                # 重新按降后的 d 分配
                a, b, reserve, d = allocate_gemm_tensors_with_retry(
                    device, dtype, d_init=d, matrices=3, max_trials=8
                )
                print(f"[GPU {gpu_id}] Re-allocated after OOM: dim={d}")

    except KeyboardInterrupt:
        pass
    finally:
        cpu_stop.set()
        cpu_proc.join(timeout=2.0)
        try:
            del a, b, reserve, h2d_buf, d2h_buf, host_f32, pinned
        except Exception:
            pass
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Stopped and cleaned up.")

# ---------------------- entry ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem-frac", type=float, default=0.80, help="单卡占用可用显存比例（更高更激进）")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--use-fp16-gemm", action="store_true", help="GPU GEMM 用 float16（更省显存、维度可更大）")
    parser.add_argument("--host-mem-gb", type=float, default=8.0, help="每进程常驻 CPU 内存（GB）")
    parser.add_argument("--cpu-dim-hint", type=int, default=8192, help="CPU GEMM 维度 hint（128 对齐）")
    parser.add_argument("--h2d-mb-per-iter", type=int, default=512, help="每轮 H2D/D2H 传输大小（MB）")
    args = parser.parse_args()

    # 提前设置一些保守环境（如用户未手动设置）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9")

    torch.cuda.init()
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    if num_gpus == 0:
        print("No GPU found. Exit.")
        return

    mp.set_start_method("spawn", force=True)
    stop_event = mp.Event()

    def handle_sig(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    procs = []
    for gid in range(num_gpus):
        p = mp.Process(
            target=occupy_gpu,
            args=(gid, args.mem_frac, args.dtype, args.host_mem_gb, args.cpu_dim_hint, args.h2d_mb_per_iter, args.use_fp16_gemm, stop_event),
        )
        p.start()
        procs.append(p)

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(1.0)
    finally:
        stop_event.set()
        for p in procs:
            p.join()

if __name__ == "__main__":
    main()
