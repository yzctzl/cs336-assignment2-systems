import time
from collections.abc import Iterator
from typing import cast

import numpy as np
import torch
from cs336_basics.config import Configures, TrainConfig
from cs336_basics.data import get_batch_iterator, load_checkpoint, save_checkpoint
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


@torch.no_grad()
def valid(model: nn.Module, valid_iter: Iterator, cfg: Configures, iters: int = 10, dtype: torch.dtype | None = None):
    """
    get 10 sample from valid set and calculate the mean loss
    """
    # change model work mode to valid/inference
    model.eval()
    losses = torch.zeros(iters)

    for k in range(iters):
        x, y = next(valid_iter)

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses[k] = loss.detach()

    # change back to train mode
    model.train()
    return losses.mean().item()


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    start_step: int,
    cfg: Configures,
    scaler: torch.GradScaler,
    train_set: np.ndarray,
    valid_set: np.ndarray,
    dtype: torch.dtype,
):
    # set the module in training mode
    model.train()

    tc: TrainConfig = cfg.train
    device = cfg.model.device

    # data iterator
    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)
    valid_iter = get_batch_iterator(valid_set, tc.batch_size, cfg.model.context_length, device)

    t0 = time.perf_counter()
    for it in range(start_step, tc.steps):
        # x, y = get_batch(train_set, tc.batch_size, cfg.model.context_length, device)
        optimizer.zero_grad(set_to_none=True)

        # use warm up + cosin lr dency schedule
        if tc.lr_scheduler == "cosine":
            lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
        if tc.lr_scheduler == "wsd":
            lr = get_lr_wsd_schedule(it, tc.lr_max, tc.lr_min, tc.steps, tc.t_w, tc.t_c)
        # update lr in all optimizer params groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # use gradient accumulation to simulate a larger batch size
        accum_loss = torch.zeros(1, device=device)
        for micro_step in range(tc.accum_steps):
            x, y = next(train_iter)

            _t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y) / tc.accum_steps
            # Wait for all kernels in all streams on a CUDA device to complete, then count time.
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            accum_loss += loss.detach()

            # Since PyTorch sums gradients in the .grad attribute by default, calling backward on each
            # micro-batch loss (scaled by 1/accum_steps) computes the average gradient for the full batch size.
            # scale the loss to prevent gradient underflow when using mixed precision (fp16).
            scaler.scale(loss).backward()
            # Wait for all kernels in all streams on a CUDA device to complete, then count time.
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            print(f"Step {it:04d} | Forward Time: {_t1 - _t0:.6f} | Backward Time: {_t2 - _t1:.6f}")

        # unscale before clipping to ensure the clipping threshold is applied to the actual gradient values.
        scaler.unscale_(optimizer)
        # clip the gradients to prevent them from exploding
        total_norm = gradient_clipping(model.parameters(), tc.grad_clip)
        # do optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        # log and save checkpoint after each interval steps
        if (it + 1) % tc.interval == 0 or it == 0:
            # Wait for all kernels in all streams on a CUDA device to complete, then count time.
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            tps = (tc.batch_size * cfg.model.context_length * tc.accum_steps * tc.interval) / dt

            # calc valid loss
            vloss = valid(model, valid_iter, cfg, dtype=dtype)
            # log to wandb
            accum_loss_ = accum_loss.item()
            print(
                f"Step {it:04d} | Time: {dt:.6f} | Valid Loss: {vloss:.4f} | Norm: {total_norm:.4f} | "
                f"Train Loss: {accum_loss_:.4f} | TPS: {tps:.1f} | LR: {lr:.2e}"
            )

            if tc.checkpoint:
                save_checkpoint(model, optimizer, it, f"./dist/checkpoint_{it:04d}_{vloss:.4f}.pt")
            t0 = time.perf_counter()


def main(
    cfg: Configures,
    fp16: bool = False,
    resume: str | None = None,
):
    """
    Benchmarking script for TinyStories models.

    Args:
        cfg: Configuration object (supports nested overrides via --cfg.train.batch_size etc.)
        fp16: Enable fp16 + grad_scaler training.
        resume: Path to a checkpoint file to resume from.
    """
    # seed
    torch.manual_seed(cfg.seed)
    # precision
    if fp16:
        dtype = torch.float16
    elif torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")
    else:
        dtype = torch.float32

    # to scale gradients and prevent underflow during float16 mixed precision training
    scaler = torch.GradScaler(cfg.model.device, enabled=(dtype == torch.float16))
    # lazy load file
    train_set = np.load(cfg.data.train)
    valid_set = np.load(cfg.data.valid)
    # init Model
    model = TransformerLM(**cfg.model.model_dump())
    model.to(cfg.model.device)
    # init Optimizer
    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    if cfg.optimizer.type == "muon":
        optimizer = Muon(model, **cfg.optimizer.model_dump())

    # load from checkpoint
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer)

    # compile the model to speedup
    model = cast(nn.Module, torch.compile(model))

    # train
    train(model, optimizer, start_step, cfg, scaler, train_set, valid_set, dtype)


if __name__ == "__main__":
    CLI(main)
