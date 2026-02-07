# pyright: reportPrivateImportUsage=none
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from torch_npu import npu
except ImportError:
    pass

def setup(rank, backend, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    if backend == "hccl":
        npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def acquire_data(size_mb, device_type):
    num_elements = (size_mb * 1024 * 1024) // 4
    device = torch.device(device_type)
    data = torch.ones(num_elements, device=device)
    return data


def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize(device)
    if device == "npu":
        npu.synchronize(device)


def benchmark_all_reduce(rank, world_size, backend, device, size_mb, results):
    # init process group
    setup(rank, backend, world_size)

    # calc number of elements
    data = acquire_data(size_mb, device)
    
    # warm up 5 times
    for _ in range(5):
        dist.all_reduce(data)
    sync_device(device)

    start_time = time.perf_counter()

    iters = 10
    for _ in range(iters):
        dist.all_reduce(data)
    sync_device(device)

    end_time = time.perf_counter()

    # Store per-rank time, aggregate later in the parent process.
    local_time = (end_time - start_time) / iters
    results[(backend, world_size, size_mb, rank)] = local_time
    if rank == 0:
        print(f"Finished: {backend.upper()} | Size: {size_mb}MB | Nodes: {world_size}")

    dist.destroy_process_group()

def aggregate_results(results):
    grouped = {}
    for key, value in results.items():
        backend, world_size, size_mb, rank = key
        grouped.setdefault((backend, world_size, size_mb), {})[rank] = value

    summary = {}
    for key, rank_times in grouped.items():
        times = list(rank_times.values())
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # All-Reduce = Reduce-Scatter + All-Gather => ~2x data volume.
        size_mb = key[2]
        world_size = key[1]
        bandwidth = (size_mb / 1024) * 2 * (world_size - 1) / world_size / avg_time
        summary[key] = (avg_time, max_time, bandwidth)
    return summary

if __name__ == "__main__":
    backends = [("hccl", "npu")]  # , ("nccl", "cuda")
    sizes = [1, 10, 100, 1000, 10000, 50000]
    world_sizes = [2, 4]  # , 6
    results = mp.Manager().dict()

    for backend, device in backends:
        for word_size in world_sizes:
            for size in sizes:
                mp.spawn(
                    benchmark_all_reduce,
                    args=(word_size, backend, device, size, results),
                    nprocs=word_size,
                    join=True
                )

    print("| Backend | Procs | Size (MB) | Avg Time (s) | Max Time (s) | Bandwidth (GB/s) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    summary = aggregate_results(results)
    for key in sorted(summary.keys()):
        avg_t, max_t, bw = summary[key]
        print(f"| {key[0].upper()} | {key[1]} | {key[2]} | {avg_t:.6f} | {max_t:.6f} | {bw:.2f} |")
