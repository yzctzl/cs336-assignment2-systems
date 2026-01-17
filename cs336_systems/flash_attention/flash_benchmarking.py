# pyright: reportArgumentType=none
import math

import pandas as pd
import torch
import triton
import triton.testing

from cs336_systems.flash_attention.flash_forward_triton import FlashAttention2Triton


def pytorch_attention(q, k, v, is_causal=True):
    """
    Standard PyTorch implementation using naive softmax attention.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        n_q, n_k = s.shape[-2:]
        mask = torch.triu(torch.ones((n_q, n_k), device=q.device), diagonal=1).bool()
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def benchmark():
    # Sequence lengths: powers of 2 from 128 to 65536
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # Embedding dimensions: powers of 2 from 16 to 128
    d_dims = [16, 32, 64, 128]
    # Precisions
    precisions = [torch.bfloat16, torch.float32]

    results = []

    for dtype in precisions:
        for d in d_dims:
            for n in seq_lengths:
                # Randomly generate inputs
                q = torch.randn((1, n, d), device="cuda", dtype=dtype, requires_grad=True)
                k = torch.randn((1, n, d), device="cuda", dtype=dtype, requires_grad=True)
                v = torch.randn((1, n, d), device="cuda", dtype=dtype, requires_grad=True)
                do = torch.randn((1, n, d), device="cuda", dtype=dtype)

                torch.cuda.reset_peak_memory_stats()

                # --- 1. Standard PyTorch Attention Benchmarking ---
                fwd_pt = float("nan")
                full_pt = float("nan")
                bwd_pt = float("nan")
                pt_status_fwd = "FAIL"
                pt_status_full = "FAIL"

                try:

                    def python_attention_fn():
                        return pytorch_attention(q, k, v, is_causal=True)

                    fwd_pt = float(triton.testing.do_bench(python_attention_fn))
                    pt_status_fwd = "OK"
                except torch.cuda.OutOfMemoryError:
                    pt_status_fwd = "OOM"
                except Exception as e:
                    pt_status_fwd = f"Error: {type(e).__name__}"

                try:
                    if pt_status_fwd == "OK":

                        def pt_full_fn():
                            o = pytorch_attention(q, k, v, is_causal=True)
                            o.backward(do, retain_graph=True)

                        full_pt = float(triton.testing.do_bench(pt_full_fn))
                        bwd_pt = full_pt - fwd_pt
                        pt_status_full = "OK"
                except torch.cuda.OutOfMemoryError:
                    pt_status_full = "OOM"
                except Exception as e:
                    pt_status_full = f"Error: {type(e).__name__}"
                
                pt_peak_memory_bytes = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()

                # --- 2. Triton Forward + Pure Backward (Partial) ---
                fwd_p_triton = float("nan")
                full_p_triton = float("nan")
                bwd_p_triton = float("nan")
                p_triton_status_fwd = "FAIL"
                p_triton_status_full = "FAIL"

                try:

                    def flash_2_attention_fn():
                        return FlashAttention2Triton.apply(q, k, v, True)

                    fwd_p_triton = float(triton.testing.do_bench(flash_2_attention_fn))
                    p_triton_status_fwd = "OK"
                except torch.cuda.OutOfMemoryError:
                    p_triton_status_fwd = "OOM"
                except Exception as e:
                    p_triton_status_fwd = f"Error: {type(e).__name__}"

                try:
                    if p_triton_status_fwd == "OK":

                        def triton_p_full_fn():
                            o = FlashAttention2Triton.apply(q, k, v, True)
                            o.backward(do, retain_graph=True)  # pyright: ignore[reportOptionalMemberAccess]

                        full_p_triton = float(triton.testing.do_bench(triton_p_full_fn))
                        bwd_p_triton = full_p_triton - fwd_p_triton
                        p_triton_status_full = "OK"
                except torch.cuda.OutOfMemoryError:
                    p_triton_status_full = "OOM"
                except Exception as e:
                    p_triton_status_full = f"Error: {type(e).__name__}"

                peak_memory_bytes = torch.cuda.max_memory_allocated()

                res = {
                    "Precision": "bf16" if dtype == torch.bfloat16 else "fp32",
                    "D": d,
                    "N": n,
                    "PT Fwd": fwd_pt,
                    "PT Bwd": bwd_pt,
                    "PT Full": full_pt,
                    "PT Mem (MB)": pt_peak_memory_bytes / (1024**2),
                    "Triton Fwd": fwd_p_triton,
                    "Pure Bwd": bwd_p_triton,
                    "Triton + Pure": full_p_triton,
                    "Peak Mem (MB)": peak_memory_bytes / (1024**2),
                }
                results.append(res)
                print(
                    f"N={n:5d}, D={d:3d}, Prec={res['Precision']} | "
                    f"PT Fwd: {fwd_pt:.3f} ({pt_status_fwd}), Triton Fwd: {fwd_p_triton:.3f} ({p_triton_status_fwd}) | "
                    f"PT Bwd: {bwd_pt:.3f} ({pt_status_full}), Pure Bwd: {bwd_p_triton:.3f} ({p_triton_status_full})"
                )

                # Clear cache to avoid memory issues for large N
                torch.cuda.empty_cache()

    # Create table
    df = pd.DataFrame(results)
    print("\nBenchmark Results Summary:")
    print(df.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    benchmark()
