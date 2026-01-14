import time
from typing import Any, cast

import numpy as np
import torch
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
from torch import nn, optim

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

WARMUP_STEPS = 5
BENCHMARK_STEPS = 10


def benchmark_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: Configures,
    train_set: np.ndarray,
    dtype: torch.dtype,
    size_name: str,
) -> dict[str, Any] | None:
    model.train()
    tc: TrainConfig = cfg.train
    device = cfg.model.device
    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)

    # skip warm up
    try:
        for _ in range(WARMUP_STEPS):
            x, y = next(train_iter)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return None

    # benchmark times
    fwd_times = []
    bwd_times = []
    opt_times = []
    step_times = []

    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(BENCHMARK_STEPS):
            x, y = next(train_iter)
            optimizer.zero_grad(set_to_none=True)

            # 1. forweard
            torch.cuda.synchronize()
            start_fwd = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y)
            torch.cuda.synchronize()
            t_fwd = time.perf_counter() - start_fwd

            # 2. backward
            start_bwd = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.perf_counter() - start_bwd

            # 3. optimizer step
            start_opt = time.perf_counter()
            gradient_clipping(model.parameters(), tc.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            t_opt = time.perf_counter() - start_opt

            # 4. collect time
            fwd_times.append(t_fwd)
            bwd_times.append(t_bwd)
            opt_times.append(t_opt)
            step_times.append(t_fwd + t_bwd + t_opt)

        memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        return {
            "fwd": np.mean(fwd_times) * 1000,
            "bwd": np.mean(bwd_times) * 1000,
            "opt": np.mean(opt_times) * 1000,
            "step": np.mean(step_times) * 1000,
            "mem": memory,
        }
    except torch.cuda.OutOfMemoryError:
        return None


def main(cfg: Configures):
    torch.manual_seed(cfg.seed)

    # print markdown format
    header = f"| {'Model':<10} | {'Mode':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Opt (ms)':<10} | {'Step (ms)':<10} | {'Mem (MB)':<10} |"
    divider = f"|:{'-' * 10}-|:{'-' * 10}-|:{'-' * 10}-|:{'-' * 10}-|:{'-' * 10}-|:{'-' * 10}-|:{'-' * 10}-|"
    print(header)
    print(divider)

    if torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")
    else:
        dtype = torch.float32

    train_set = np.load(cfg.data.train)

    for size_name in ["small", "medium", "large", "xl", "2.7B"]:
        params = MODEL_SIZES[size_name]
        cfg.model.d_model = params["d_model"]
        cfg.model.d_ff = params["d_ff"]
        cfg.model.num_layers = params["num_layers"]
        cfg.model.num_heads = params["num_heads"]

        # compare eager mode and complied mode
        for is_compile in [False, True]:
            mode_str = "Compiled" if is_compile else "Eager"

            try:
                model = TransformerLM(**cfg.model.model_dump())
                if is_compile:
                    model = cast(nn.Module, torch.compile(model))
                model.to(cfg.model.device)

                if cfg.optimizer.type == "adamw":
                    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
                else:
                    optimizer = Muon(model, **cfg.optimizer.model_dump())

                metrics = benchmark_train(model, optimizer, cfg, train_set, dtype, size_name)

                if metrics:
                    print(
                        f"| {size_name:<10} | {mode_str:<10} | {metrics['fwd']:<10.2f} | {metrics['bwd']:<10.2f} | {metrics['opt']:<10.2f} | {metrics['step']:<10.2f} | {metrics['mem']:<10.2f} |"
                    )
                else:
                    print(
                        f"| {size_name:<10} | {mode_str:<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} |"
                    )

                # clean up old memory
                del model, optimizer
                torch.cuda.empty_cache()

            except Exception as e:
                print(
                    f"| {size_name:<10} | {mode_str:<10} | Error: {str(e)[:20]:<20} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} |"
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    CLI(main)
