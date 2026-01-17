import time

import numpy as np
import pandas as pd
import torch
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
from torch import nn, optim

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
result = []


def benchmark_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: Configures,
    train_set: np.ndarray,
    dtype: torch.dtype,
    size_name: str,
):
    model.train()

    tc: TrainConfig = cfg.train
    device = cfg.model.device

    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)

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

            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y)
            torch.cuda.synchronize()

            t1 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()

            t2 = time.perf_counter()
            gradient_clipping(model.parameters(), tc.grad_clip)
            optimizer.step()
            t3 = time.perf_counter()

            if it > 0:
                result.append({
                    "Model": size_name,
                    "Context": cfg.model.context_length,
                    "Forward(ms)": (t1-t0) * 1000,
                    "Backward(ms)": (t2-t1) * 1000,
                    "Step(ms)": (t3-t2) * 1000,
                    "Total(ms)": (t3-t0) * 1000,
                })

    except torch.cuda.OutOfMemoryError:
        print(f"⚠️ {size_name}:{cfg.model.context_length} OOM!")
        result.append({
                    "Model": size_name,
                    "Context": cfg.model.context_length,
                    "Forward(ms)": np.nan,
                    "Backward(ms)": np.nan,
                    "Step(ms)": np.nan,
                    "Total(ms)": np.nan
                })
        return False


def main(cfg: Configures):
    """
    Benchmarking script for TinyStories models.

    Args:
        cfg: Configuration object (supports nested overrides via --cfg.train.batch_size etc.)
        fp16: Enable fp16 + grad_scaler training.
        resume: Path to a checkpoint file to resume from.
    """

    torch.manual_seed(cfg.seed)
    if torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")
    else:
        dtype = torch.float32

    train_set = np.load(cfg.data.train)

    for size_name, params in MODEL_SIZES.items():
        print(f"Benchmarking {size_name}...")

        cfg.model.d_model = params["d_model"]
        cfg.model.d_ff = params["d_ff"]
        cfg.model.num_layers = params["num_layers"]
        cfg.model.num_heads = params["num_heads"]

        model = TransformerLM(**cfg.model.model_dump())
        model.to(cfg.model.device)

        if cfg.optimizer.type == "adamw":
            optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
        else:
            optimizer = Muon(model, **cfg.optimizer.model_dump())

        benchmark_train(model, optimizer, cfg, train_set, dtype, size_name)

        del model, optimizer
        torch.cuda.empty_cache()

    model_order = list(MODEL_SIZES.keys())
    df = pd.DataFrame(result)
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    summary = df.groupby(["Model", "Context"], observed=True).agg({
        "Forward(ms)": "mean",
        "Backward(ms)": "mean",
        "Step(ms)": "mean",
        "Total(ms)": ["mean", "std"],
    })
    summary.columns = [
        "Forward(ms)", 
        "Backward(ms)", 
        "Step(ms)", 
        "Total(ms)", 
        "Total_Std",
    ]
    summary = summary.reset_index()
    summary["CV(%)"] = (summary["Total_Std"] / summary["Total(ms)"]) * 100

    final_cols = [
        "Model", "Context", "Forward(ms)", "Backward(ms)", 
        "Step(ms)", "Total(ms)", "CV(%)"
    ]

    print("\n## Benchmark Summary (ms)")
    print(summary[final_cols].to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    CLI(main)
