from contextlib import nullcontext
import numpy as np
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
    # "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    # "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    # "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    # "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
result = []


def benchmark_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: Configures,
    train_set: np.ndarray,
    dtype: torch.dtype
):
    model.train()

    tc: TrainConfig = cfg.train
    device = cfg.model.device

    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)
    ctx = torch.autocast(device_type=device, dtype=dtype) if dtype != torch.float32 else nullcontext()

    for it in range(tc.steps):
        optimizer.zero_grad(set_to_none=True)

        if tc.lr_scheduler == "cosine":
            lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
        if tc.lr_scheduler == "wsd":
            lr = get_lr_wsd_schedule(it, tc.lr_max, tc.lr_min, tc.steps, tc.t_w, tc.t_c)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = next(train_iter)

        if it == 5:
            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        with ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)

        if it == 5:
            torch.cuda.synchronize()
            # Save a pickle file to be loaded by PyTorch's online tool.
            torch.cuda.memory._dump_snapshot(f"./dist/large_forward_snapshot_{cfg.model.context_length}.pickle")

        loss.backward()

        if it == 5:
            torch.cuda.synchronize()
            # Save a pickle file to be loaded by PyTorch's online tool.
            torch.cuda.memory._dump_snapshot(f"./dist/large_backward_snapshot_{cfg.model.context_length}.pickle")

        gradient_clipping(model.parameters(), tc.grad_clip)
        optimizer.step()

        if it == 5:
            torch.cuda.synchronize()
            # Save a pickle file to be loaded by PyTorch's online tool.
            torch.cuda.memory._dump_snapshot(f"./dist/large_full_snapshot_{cfg.model.context_length}.pickle")
            # Stop recording history.
            torch.cuda.memory._record_memory_history(enabled=None)


def main(cfg: Configures, fp32: bool = False):
    torch.manual_seed(cfg.seed)

    if fp32:
        dtype = torch.float32
    elif torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")

    train_set = np.load(cfg.data.train)

    for context_length in (128, 256, 512):
        cfg.model.context_length = context_length

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

            try:
                benchmark_train(model, optimizer, cfg, train_set, dtype)
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️ {size_name}:{cfg.model.context_length} OOM!")
                return False

            del model, optimizer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    CLI(main)
