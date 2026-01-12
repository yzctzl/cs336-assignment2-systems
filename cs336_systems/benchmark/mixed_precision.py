import numpy as np
import torch
from cs336_basics.data import get_batch_iterator
from cs336_basics.optimizer import (
    AdamW,
    cross_entropy,
    gradient_clipping,
)
from torch import nn, optim


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.RMSNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dtype == torch.long:
            x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x


def benchmark_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.GradScaler,
    dtype: torch.dtype,
    train_iter,
):
    model.train()
    device = "cuda"

    for it in range(10):
        optimizer.zero_grad(set_to_none=True)

        x, y = next(train_iter)

        with torch.autocast(device_type=device, dtype=dtype):
            logits = model(x)
            loss = cross_entropy(logits, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        gradient_clipping(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


def main(fp16: bool = False, checkpoint_path: str = None):
    if fp16:
        dtype = torch.float16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")
    else:
        dtype = torch.float32

    scaler = torch.GradScaler("cuda", enabled=(dtype == torch.float16))

    model = ToyModel(768, 3072)
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    model.to("cuda")

    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    train_set = np.random.randint(0, 3072, (1000, 768))
    train_iter = get_batch_iterator(train_set, batch_size=4, context_length=768, device="cuda")

    benchmark_train(model, optimizer, scaler, dtype, train_iter)


if __name__ == "__main__":
    main()
