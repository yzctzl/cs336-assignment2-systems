import math

import numpy as np
import torch
from cs336_basics import model
from cs336_basics.config import Configures, TrainConfig
from cs336_basics.data import get_batch_iterator
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.optimizer import (
    AdamW,
    Muon,
    cross_entropy,
    get_lr_cosine_schedule,
    get_lr_wsd_schedule,
    gradient_clipping,
)
from einops import einsum
from jaxtyping import Bool, Float
from jsonargparse import CLI
from torch import Tensor, nn, optim
from torch.cuda import nvtx


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_k"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "... seq_len d_k"]:
    """
    The scaled dot-product attention function

    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    """
    d_k = Q.shape[-1]
    d_type = Q.dtype
    with nvtx.range("computing attention scores"):
        # attention score = QK^T / sqrt(d_k)
        # in tensor core, fp16 matmul use fp32 alu, so it's safe with fp16
        scores_ = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        # must cast to float32 for numerical stability before scaling and softmax to 
        # prevent potential overflow or precision loss when inputs are in float16.
        if scores_.dtype in (torch.float16, torch.bfloat16):
            scores_ = scores_.to(torch.float32)
        scores = scores_ / math.sqrt(d_k)
    with nvtx.range("computing softmax"):
        # it will be much more efficient to use masking than to compute attention on subsequences
        if mask is not None:
            masked = torch.where(mask, scores, float("-inf"))
        else:
            masked = scores
        # apply softmax to scores -1 dim (keys) for each individual query (dim -2)
        # to normalize the scores into a distribution over all keys for each query
        weights = softmax(masked, -1).to(d_type)
    with nvtx.range("final matmul"):
        # Attention(Q, K, V) = softmax(scores) @ V
        att = einsum(weights, V, "... queries seq_len, ... seq_len d_k -> ... queries d_k")
    return att


model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


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

            with nvtx.range("forward pass"):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = cross_entropy(logits, y)

            with nvtx.range("backward pass"):
                loss.backward()

            with nvtx.range("optimizer step"):
                gradient_clipping(model.parameters(), tc.grad_clip)
                optimizer.step()

    except torch.cuda.OutOfMemoryError:
        print(f"⚠️ {size_name}:{cfg.model.context_length} OOM!")
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


if __name__ == "__main__":
    CLI(main)
