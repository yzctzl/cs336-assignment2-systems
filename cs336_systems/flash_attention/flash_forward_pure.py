import math

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch.types import Tensor

B_q = 64  # tile size of Q
B_k = 64  # tile size of K, V


class FlashAttention2Pure(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_k"],
        is_causal=False,
    ):
        # record the original shape, compatible 3d/4d/5d
        N, d = Q.shape[-2:]
        device = Q.device
        dtype = Q.dtype

        # flatten Q/K/V for apply Algorithm 1
        Q_flat = rearrange(Q, "... N d -> (...) N d")  # (merged_B seq_len d_k)
        K_flat = rearrange(K, "... N d -> (...) N d")
        V_flat = rearrange(V, "... N d -> (...) N d")
        merged_B = Q_flat.shape[0]

        # tile count of Q, K/V
        T_q = (N + B_q - 1) // B_q
        T_k = (N + B_k - 1) // B_k

        # init the return value: Output and LogSumExp
        O_flat = torch.zeros_like(Q_flat)
        LSE_flat = torch.zeros((merged_B, N), device=device, dtype=dtype)

        scale = 1.0 / math.sqrt(d)

        # Outer loop over Q
        for i in range(T_q):
            # load Q_i
            start_q, end_q = i * B_q, min(N, (i + 1) * B_q)
            Q_i = Q_flat[:, start_q:end_q, :]

            # initializate output, logsumexp and maximum so far of this tile
            O_i = torch.zeros_like(Q_i)
            lse_i = torch.zeros((merged_B, end_q - start_q), device=device, dtype=dtype)
            m_i = torch.full((merged_B, end_q - start_q), -torch.inf, device=device, dtype=dtype)

            # Inner loop over K, V tiles
            for j in range(T_k):
                # load K_j, V_j
                start_k, end_k = j * B_k, min(N, (j + 1) * B_k)
                K_j = K_flat[:, start_k:end_k, :]
                V_j = V_flat[:, start_k:end_k, :]

                # compute pre-softmax scores: Sij = (Qi @ Kj^T) / sqrt(d)
                s_ij = einsum(Q_i, K_j, "B B_q d, B B_k d -> B B_q B_k") * scale

                # Online Softmax
                # 1. Get the max score for the current tile (row-wise)
                m_ij = torch.max(s_ij, dim=-1).values
                # 2. Update the global max seen so far for each row in the Q-tile
                _m_i = torch.max(m_i, m_ij)

                # 3. Compute local exponentiated scores using the NEW global max
                p_ij = torch.exp(s_ij - _m_i.unsqueeze(-1))

                # 4. Rescale the previous accumulated statistics to the NEW global max
                # This ensures numerical stability by keeping exponents small
                rescale_factor = torch.exp(m_i - _m_i)

                # 5. Update the running sum of exponents (denominator of softmax)
                lse_i = rescale_factor * lse_i + torch.sum(p_ij, dim=-1)

                # 6. Update the running weighted sum of values (numerator of attention)
                # Rescale old O_i and add the contribution from the current tile
                O_i = rescale_factor.unsqueeze(-1) * O_i + torch.matmul(p_ij, V_j)

                # 7. Move to the next iteration with the new global max
                m_i = _m_i

            # Final normalization for the Q-tile: divide by the sum of exponents
            O_flat[:, start_q:end_q, :] = O_i / lse_i.unsqueeze(-1)
            # LogSumExp is stored for the backward pass: L = m + log(l)
            LSE_flat[:, start_q:end_q] = m_i + torch.log(lse_i)

        Output = O_flat.view(*Q.shape)
        LSE = LSE_flat.view(*Q.shape[:-1])

        ctx.save_for_backward(Q, K, V, Output, LSE)
        ctx.is_causal = is_causal

        return Output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        raise NotImplementedError
