import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.config import Configures, TrainConfig
from cs336_basics.data import get_batch_iterator
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
    AdamW,
    Muon,
    cross_entropy,
    gradient_clipping,
)
from jsonargparse import CLI

# Adapters
from tests.adapters import (
    ddp_bucketed_on_after_backward,
    ddp_bucketed_on_train_batch_start,
    ddp_individual_parameters_on_after_backward,
    get_ddp_bucketed,
    get_ddp_individual_parameters,
)
from tests.common import (
    _cleanup_process_group,
    _empty_cache,
    _get_backend,
    _setup_process_group,
    _sync_device,
)

MODEL_SIZES = {
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def benchmark_ddp_train(
    rank: int,
    world_size: int,
    backend: str,
    cfg: Configures,
    train_set: np.ndarray,
    ddp_type: str,
    bucket_size_mb: float,
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()
    torch.manual_seed(cfg.seed + rank)

    # Initialize model
    model = TransformerLM(**cfg.model.model_dump())
    model.to(device)

    # Wrap with DDP
    if ddp_type == "individual":
        model = get_ddp_individual_parameters(model)
    elif ddp_type == "bucketed":
        model = get_ddp_bucketed(model, bucket_size_mb=bucket_size_mb)
    else:
        raise ValueError(f"Unknown ddp_type: {ddp_type}")

    # Optimizer
    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    else:
        optimizer = Muon(model, **cfg.optimizer.model_dump())

    model.train()
    tc: TrainConfig = cfg.train
    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)

    times = []

    try:
        # Warmup
        for i in range(5):
            # Ensure data iterator doesn't run out for small datasets
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)
                x, y = next(train_iter)

            if ddp_type == "bucketed":
                ddp_bucketed_on_train_batch_start(model, optimizer)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()

            if ddp_type == "individual":
                ddp_individual_parameters_on_after_backward(model, optimizer)
            elif ddp_type == "bucketed":
                ddp_bucketed_on_after_backward(model, optimizer)

            optimizer.step()
            _sync_device(device)

        # Benchmark
        for it in range(tc.steps):
            if ddp_type == "bucketed":
                ddp_bucketed_on_train_batch_start(model, optimizer)

            optimizer.zero_grad(set_to_none=True)

            # LR Schedule (simplified from original)
            lr = tc.lr_max
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)
                x, y = next(train_iter)

            t1 = time.perf_counter()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()

            t2 = time.perf_counter()  # End of backward (includes communication for overlapping)

            if ddp_type == "individual":
                ddp_individual_parameters_on_after_backward(model, optimizer)
            elif ddp_type == "bucketed":
                ddp_bucketed_on_after_backward(model, optimizer)

            t3 = time.perf_counter()  # End of synchronization waiting

            gradient_clipping(model.parameters(), tc.grad_clip)
            optimizer.step()
            _sync_device(device)
            t4 = time.perf_counter()

            total_step_time = t4 - t1
            times.append(total_step_time)

            if rank == 0:
                print(f"Step {it:2d} | Total: {total_step_time:.3f}s")

        if rank == 0:
            avg_time = sum(times) / len(times)
            print(f"\nAverage Step Time ({ddp_type}, bucket={bucket_size_mb}): {avg_time:.4f}s")

            # Write results to a file for easy reading
            with open("benchmark_results_raw.txt", "a") as f:
                f.write(f"{ddp_type},{bucket_size_mb},{avg_time:.4f}\n")

    except torch._C.OutOfMemoryError:
        print(f"⚠️ {cfg.model.context_length} OOM!")
        return False
    finally:
        del model, optimizer
        _empty_cache()
        _cleanup_process_group()


def benchmark_runner(cfg: Configures, train_set, backend, ddp_type, bucket_size_mb=None):
    world_size = 2
    import os
    import random

    os.environ["MASTER_PORT"] = str(random.randint(20000, 60000))

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        benchmark_ddp_train,
        args=(world_size, backend, cfg, train_set, ddp_type, bucket_size_mb),
        nprocs=world_size,
        join=True,
    )


def main(cfg: Configures):
    train_set = np.load(cfg.data.train)
    backend = _get_backend()
    print(f"Using backend: {backend}")

    params = MODEL_SIZES["xl"]
    cfg.model.d_model = params["d_model"]
    cfg.model.d_ff = params["d_ff"]
    cfg.model.num_layers = params["num_layers"]
    cfg.model.num_heads = params["num_heads"]

    # 1. Benchmark Individual
    print("\n=== Benchmarking Individual Overlap ===")
    benchmark_runner(cfg, train_set, backend, "individual")

    # 2. Benchmark Bucketed with various sizes
    bucket_sizes = [1, 10, 100, 1000]
    for b_size in bucket_sizes:
        print(f"\n=== Benchmarking Bucketed Overlap (Size: {b_size}MB) ===")
        benchmark_runner(cfg, train_set, backend, "bucketed", b_size)


if __name__ == "__main__":
    CLI(main)
