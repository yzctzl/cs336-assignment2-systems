import time

import torch
from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
DEVICE = "cuda"
WARMUP = 10
STEPS = 100


compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)


def benchmark(d_model: int, seq_len: int):
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)

    for _ in range(WARMUP):
        out = compiled_scaled_dot_product_attention(Q, K, V)
        out.sum().backward()
        torch.cuda.synchronize()

    start_f = time.perf_counter()
    for _ in range(STEPS):
        out = compiled_scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
    end_f = time.perf_counter()
    forward_time = (end_f - start_f) / STEPS

    out = compiled_scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    memory_usage = torch.cuda.memory_allocated() / (1024**2)  # MB

    start_total = time.perf_counter()
    for _ in range(STEPS):
        out = compiled_scaled_dot_product_attention(Q, K, V)
        out.sum().backward()
        torch.cuda.synchronize()
    end_total = time.perf_counter()
    total_step_time = (end_total - start_total) / STEPS
    backward_time = max(0.0, total_step_time - forward_time)

    return forward_time, backward_time, memory_usage


def main():
    torch.set_float32_matmul_precision("high")
    print(f"{'d_model':<10} | {'seq_len':<10} | {'Forward (ms)':<15} | {'Backward (ms)':<15} | {'Memory (MB)':<12}")
    print("-" * 75)

    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    for d_model in d_models:
        for context_length in seq_lengths:
            try:
                f_time, b_time, mem = benchmark(d_model, context_length)
                print(
                    f"{d_model:<10} | {context_length:<10} | {f_time * 1000:<15.4f} | {b_time * 1000:<15.4f} | {mem:<12.2f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{d_model:<10} | {context_length:<10} | {'OOM':<15} | {'OOM':<15} | {'OOM':<12}")
                torch.cuda.empty_cache()
            finally:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
