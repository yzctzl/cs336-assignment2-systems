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
    def backward(
        ctx,
        grad_out: Float[Tensor, "... N_q d"],
    ):
        """
        Pure backward without tiling
        """
        Q: torch.Tensor
        K: torch.Tensor
        V: torch.Tensor
        Output: torch.Tensor
        LSE: torch.Tensor
        Q, K, V, Output, LSE = ctx.saved_tensors
        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # 1. Recompute attention scores S = (Q @ K^T) * scale
        S = einsum(Q, K, "... N_q d, ... N_k d -> ... N_q N_k") * scale

        # 2. Apply causal mask if it was used in forward pass
        if ctx.is_causal:
            N_q, N_k = S.shape[-2:]
            mask = torch.ones((N_q, N_k), device=S.device, dtype=torch.bool).tril(diagonal=N_k - N_q)
            S = torch.where(mask, S, float("-inf"))

        # 3. Recompute softmax probabilities P = exp(S - LSE)
        # Using LSE ensures numerical stability similar to the log-sum-exp trick
        P = torch.exp(S - LSE.unsqueeze(-1))

        # 4. Compute gradient for V: grad_V = P^T @ grad_out
        grad_V = einsum(P, grad_out, "... N_q N_k, ... N_q d -> ... N_k d")

        # 5. Compute intermediate gradient for probabilities: grad_P = grad_out @ V^T
        grad_P = einsum(grad_out, V, "... N_q d, ... N_k d -> ... N_q N_k")

        # 6. Compute D term: D = rowsum(Output * grad_out)
        # In FlashAttention-2, D helps simplify the softmax gradient calculation
        D = torch.sum(Output * grad_out, dim=-1)

        # 7. Compute gradient for scores S: grad_S = P * (grad_P - D)
        # This is a fused version of the standard softmax backward pass
        grad_S = P * (grad_P - D.unsqueeze(-1))

        # 8. Compute gradients for Q and K
        grad_Q = einsum(grad_S, K, "... N_q N_k, ... N_k d -> ... N_q d") * scale
        grad_K = einsum(grad_S, Q, "... N_q N_k, ... N_q d -> ... N_k d") * scale

        # The last None is for the is_causal argument which doesn't require a gradient
        return grad_Q, grad_K, grad_V, None
