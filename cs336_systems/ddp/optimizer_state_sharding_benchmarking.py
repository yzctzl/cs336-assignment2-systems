# pyright: reportAttributeAccessIssue=none
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

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

try:
    from torch_npu import npu  # type: ignore
    HAS_NPU = True
except Exception:
    HAS_NPU = False

from cs336_systems.ddp.optimizer_state_sharding import ShardedOptimizer

MODEL_SIZES = {
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


@dataclass
class MemStats:
    allocated_mb: float
    max_allocated_mb: float


def _mem_stats(device: str) -> Optional[MemStats]:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return MemStats(allocated, max_allocated)
    if device.startswith("npu") and HAS_NPU:
        npu.synchronize()
        allocated = npu.memory_allocated() / (1024 ** 2)
        max_allocated = npu.max_memory_allocated() / (1024 ** 2)
        return MemStats(allocated, max_allocated)
    return None


def _set_lr(optimizer, tc: TrainConfig, it: int):
    if tc.lr_scheduler == "cosine":
        lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
    elif tc.lr_scheduler == "wsd":
        lr = get_lr_wsd_schedule(it, tc.lr_max, tc.lr_min, tc.steps, tc.t_w, tc.t_c)
    else:
        lr = tc.lr_max
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _make_optimizer(model: torch.nn.Module, cfg: Configures, sharded: bool):
    if cfg.optimizer.type == "adamw":
        if sharded:
            return ShardedOptimizer(model.parameters(), AdamW, **cfg.optimizer.model_dump())
        return AdamW(model.parameters(), **cfg.optimizer.model_dump())
    if sharded:
        return ShardedOptimizer(model.parameters(), Muon, **cfg.optimizer.model_dump())
    return Muon(model, **cfg.optimizer.model_dump())


def _benchmark_rank(
    rank: int,
    world_size: int,
    backend: str,
    cfg: Configures,
    train_set: np.ndarray,
    sharded: bool,
    warmup_steps: int,
    measure_steps: int,
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()

    torch.manual_seed(cfg.seed)

    model = TransformerLM(**cfg.model.model_dump()).to(device)
    optimizer = _make_optimizer(model, cfg, sharded=sharded)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if device.startswith("npu") and HAS_NPU:
        npu.reset_peak_memory_stats()

    mem_init = _mem_stats(device)

    model.train()
    tc: TrainConfig = cfg.train
    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)

    for it in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        _set_lr(optimizer, tc, it)
        x, y = next(train_iter)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), tc.grad_clip)
        optimizer.step()
        _sync_device(device)

    times = []
    mem_pre_step = None
    mem_post_step = None

    for it in range(measure_steps):
        optimizer.zero_grad(set_to_none=True)
        _set_lr(optimizer, tc, it)
        x, y = next(train_iter)

        t0 = time.perf_counter()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        _sync_device(device)

        if it == 0:
            mem_pre_step = _mem_stats(device)

        gradient_clipping(model.parameters(), tc.grad_clip)
        optimizer.step()
        _sync_device(device)

        if it == 0:
            mem_post_step = _mem_stats(device)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)

    if rank == 0:
        tag = "sharded" if sharded else "baseline"
        print(f"[{tag}] avg_step_time_s={avg_time:.6f}")
        if mem_init is None:
            print(f"[{tag}] mem: N/A (no cuda/npu)")
        else:
            print(
                f"[{tag}] mem_init_alloc_mb={mem_init.allocated_mb:.2f} mem_init_peak_mb={mem_init.max_allocated_mb:.2f}"
            )
            if mem_pre_step is not None:
                print(
                    f"[{tag}] mem_pre_step_alloc_mb={mem_pre_step.allocated_mb:.2f} mem_pre_step_peak_mb={mem_pre_step.max_allocated_mb:.2f}"
                )
            if mem_post_step is not None:
                print(
                    f"[{tag}] mem_post_step_alloc_mb={mem_post_step.allocated_mb:.2f} mem_post_step_peak_mb={mem_post_step.max_allocated_mb:.2f}"
                )

    del model, optimizer
    _empty_cache()
    _cleanup_process_group()


def run_benchmark(
    cfg: Configures,
    warmup_steps: int = 2,
    measure_steps: int = 5,
    use_sharded: bool = True,
):
    train_set = np.load(cfg.data.train)
    backend = _get_backend()
    world_size = 2

    os.environ["MASTER_PORT"] = str(random.randint(20000, 60000))
    mp.spawn(
        _benchmark_rank,
        args=(world_size, backend, cfg, train_set, use_sharded, warmup_steps, measure_steps),
        nprocs=world_size,
        join=True,
    )


def main(cfg: Configures, warmup_steps: int = 2, measure_steps: int = 5):
    cfg.model.context_length = cfg.model.context_length or 128

    params = MODEL_SIZES["xl"]
    cfg.model.d_model = params["d_model"]
    cfg.model.d_ff = params["d_ff"]
    cfg.model.num_layers = params["num_layers"]
    cfg.model.num_heads = params["num_heads"]

    print("=== Baseline (no sharding) ===")
    run_benchmark(cfg, warmup_steps=warmup_steps, measure_steps=measure_steps, use_sharded=False)

    print("=== Sharded Optimizer ===")
    run_benchmark(cfg, warmup_steps=warmup_steps, measure_steps=measure_steps, use_sharded=True)


if __name__ == "__main__":
    CLI(main)
