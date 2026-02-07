import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from cs336_basics.config import Configures, TrainConfig
from cs336_basics.data import get_batch_iterator
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
    AdamW,
    Muon,
    cross_entropy,
    get_lr_cosine_schedule,
    get_lr_wsd_schedule,
    gradient_clipping,
)
from jsonargparse import CLI

from tests.common import (
    _cleanup_process_group,
    _empty_cache,
    _get_backend,
    _setup_process_group,
    _sync_device,
)

MODEL_SIZES = {
    # "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    # "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    # "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    # "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def sync_ddp_parameters(model: nn.Module, src: int = 0):
    state_dict = model.state_dict()
    keys = sorted(state_dict.keys())
    for key in keys:
        tensor = state_dict[key]
        dist.broadcast(tensor, src=src)
    # dist.barrier()


@torch.no_grad()
def ddp_after_backward(ddp_model: nn.Module):
    for param in ddp_model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, dist.ReduceOp.AVG)


def benchmark_ddp_train(
    rank: int,
    world_size: int,
    backend: str,
    cfg: Configures,
    train_set: np.ndarray,
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    torch.manual_seed(cfg.seed + rank)

    model = TransformerLM(**cfg.model.model_dump())
    model.to(device)

    sync_ddp_parameters(model)

    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    else:
        optimizer = Muon(model, **cfg.optimizer.model_dump())

    model.train()

    tc: TrainConfig = cfg.train

    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)

    time_records = []
    try:
        for it in range(tc.steps):
            optimizer.zero_grad(set_to_none=True)

            if tc.lr_scheduler == "cosine":
                lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
            if tc.lr_scheduler == "wsd":
                lr = get_lr_wsd_schedule(it, tc.lr_max, tc.lr_min, tc.steps, tc.t_w, tc.t_c)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = next(train_iter)

            t1 = time.perf_counter()
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            _sync_device(device)

            t2 = time.perf_counter()
            ddp_after_backward(model)

            t3 = time.perf_counter()
            gradient_clipping(model.parameters(), tc.grad_clip)
            optimizer.step()
            _sync_device(device)
            t4 = time.perf_counter()
            time_records.append((t1, t2, t3, t4))

            total_step_time = t4 - t1
            comm_time = t3 - t2
            comm_percent = (comm_time / total_step_time) * 100

            print(
                f"Step {it:2d} | Total: {total_step_time:.3f}s | "
                f"Fwd/Bwd: {t2 - t1:.3f}s | "
                f"Comm (All-Reduce): {comm_time:.3f}s ({comm_percent:.1f}%) | "
                f"Opt: {t4 - t3:.3f}s"
            )

        if rank == 0:
            avg_time = sum([t4 - t1 for t1, t2, t3, t4 in time_records]) / len(time_records)
            print(f"\nAverage Step Time (Naive DDP): {avg_time:.4f}s")
            with open("benchmark_results_raw.txt", "a") as f:
                f.write(f"naive_ddp,N/A,{avg_time:.4f}\n")

    except torch._C.OutOfMemoryError:
        print(f"⚠️ {cfg.model.context_length} OOM!")
        return False
    finally:
        del model, optimizer
        _empty_cache()
        _cleanup_process_group()


def benchmark_naive_ddp(cfg: Configures, train_set, backend):
    world_size = 2
    os.environ["MASTER_PORT"] = str(random.randint(20000, 60000))
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        benchmark_ddp_train,
        args=(world_size, backend, cfg, train_set),
        nprocs=world_size,
        join=True,
    )


def main(cfg: Configures):
    train_set = np.load(cfg.data.train)
    backend = _get_backend()

    for size_name, params in MODEL_SIZES.items():
        print(f"Benchmarking {size_name}...")

        cfg.model.d_model = params["d_model"]
        cfg.model.d_ff = params["d_ff"]
        cfg.model.num_layers = params["num_layers"]
        cfg.model.num_heads = params["num_heads"]

        benchmark_naive_ddp(cfg, train_set, backend=backend)  # or nccl


if __name__ == "__main__":
    CLI(main)
