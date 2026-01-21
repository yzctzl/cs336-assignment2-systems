# pyright: reportPrivateImportUsage=none
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, backend, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def acquire_data(size_mb, device_type):
    num_elements = (size_mb * 1024 * 1024) // 4
    device = torch.device(device_type)
    data = torch.ones(num_elements, device=device)
    return data


def sync_cuda(device):
    if device == "cuda":
        torch.cuda.synchronize(device)


def benchmark_all_reduce(rank, world_size, backend, device, size_mb, results):
    # init process group
    setup(rank, backend, world_size)

    # calc number of elements
    data = acquire_data(size_mb, device)
    
    # warm up 5 times
    for _ in range(5):
        dist.all_reduce(data)
    sync_cuda(device)

    start_time = time.perf_counter()

    iters = 10
    for _ in range(iters):
        dist.all_reduce(data)
    sync_cuda(device)

    end_time = time.perf_counter()
    local_time = torch.tensor([end_time - start_time], device=device)

    all_times = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_times, local_time)

    all_times_list = [t.item() / iters for t in all_times]
    avg_time = sum(all_times_list) / len(all_times_list)

    if rank == 0:
        # All-Reduce = Recude-Scatter + All-Gather = 2 * params
        bandwidth = (size_mb / 1024) * 2 * (world_size - 1) / world_size / avg_time
        results[(backend, world_size, size_mb)] = (avg_time, bandwidth, all_times_list)
        print(f"Finished: {backend.upper()} | Size: {size_mb}MB | Nodes: {world_size}")

    dist.destroy_process_group()

if __name__ == "__main__":
    backends = [("gloo", "cpu")]  # , ("nccl", "cuda")
    sizes = [1, 10, 100, 1000]
    world_sizes = [2, 4 , 6]  # , 6
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

    for key in sorted(results.keys()):
        avg_t, bw, all_ts = results[key]
        max_t = max(all_ts)
        print(f"| {key[0].upper()} | {key[1]} | {key[2]} | {avg_t:.6f} | {max_t:.6f} | {bw:.2f} |")
