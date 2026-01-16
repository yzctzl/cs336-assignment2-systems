# ruff: noqa: F841, E741
# pyright: reportUnreachable=false
import math

import torch
import triton
from einops import rearrange
from jaxtyping import Float
from torch.types import Tensor
from triton import language as tl

QTS = 32  # tile size of Q
KTS = 32  # tile size of K, V


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # row indices for the current Q tile
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # make_block_ptr: returns a pointer to a block in a parent tensor
    #   base – The base pointer to the parent tensor
    #   shape – The shape of the parent tensor
    #   strides – The strides of the parent tensor
    #   offsets – The offsets to the block
    #   block_shape – The shape of the block
    #   order – The order of the original data format
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # launch grid (batch_size, q_tiles)
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # in a batch, find tiles for this thread
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # for all K/V in a batch, so start from 0
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),  # L is 2D: (Batch_size, N_Queries)
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    m = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)  # don't load from hbm, init in register file
    o = tl.zeros((Q_TILE_SIZE, D), tl.float32)  # register file(RF) is larger/faster than SRAM

    # each thread run with a tile of Q, load: HBM -> SRAM(Shared Mem) -> Register File(RF)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # bf16

    # for 0 <= i < num_k_tiles
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load K(j), V(j) from global memory
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").T
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Tensor Core only accept 16b operand, convert to bf16
        # Compute tile of pre-softmax attention scores
        s = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)) * scale

        # Causal Attention
        if is_causal:
            # index of k current tile
            k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # causal mask: query at index i can only attend to keys at index j <= i
            mask = q_indices[:, None] >= k_indices[None, :]
            s = tl.where(mask, s, float("-inf"))

        # Online Softmax
        # Update the global max seen so far for each row in the Q-tile
        _m = tl.maximum(m, tl.max(s, axis=-1))
        # Compute local exponentiated scores using the NEW global max
        p = tl.exp(s - _m[:, None])
        # Rescale the previous accumulated statistics to the NEW global max
        rescale = tl.exp(m - _m)
        # Update the running sum of exponents (denominator of softmax)
        l = rescale * l + tl.sum(p, axis=-1)
        # Update the running weighted sum of values (numerator of attention)
        o = o * rescale[:, None]
        o = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=o)

        # update global max
        m = _m

        # Move the pointers to the next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))  # advance in dim N
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # compute the final output and lse
    o = o * (1.0 / l)[:, None]
    l = m + tl.log(l)

    # save to HBM
    tl.store(O_block_ptr, o.to(q.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_k"],
        is_causal=False,
    ):
        # record the original shape, compatible 3d/4d/5d
        *B, N_Q, d = Q.shape
        N_K = K.shape[-2]
        device = Q.device
        dtype = Q.dtype

        # check QKV is in cuda
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"

        # flatten Q/K/V for apply Algorithm 1
        Q_flat = rearrange(Q, "... N d -> (...) N d").contiguous()  # (merged_B seq_len d_k)
        K_flat = rearrange(K, "... N d -> (...) N d").contiguous()
        V_flat = rearrange(V, "... N d -> (...) N d").contiguous()
        m_B = Q_flat.shape[0]  # get merged B

        # tile count of Q, K/V
        T_q = (N_Q + QTS - 1) // QTS
        # T_k = (N_K + KTS - 1) // KTS

        # init the return value: Output and LogSumExp, need float32 for LSE
        O_flat = torch.zeros_like(Q_flat, device=device, dtype=dtype)
        L_flat = torch.zeros((m_B, N_Q), device=device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)

        ctx.D = d
        ctx.Q_TILE_SIZE = QTS
        ctx.K_TILE_SIZE = KTS
        flash_fwd_kernel[(T_q, m_B)](
            Q_flat, K_flat, V_flat,
            O_flat, L_flat,
            *Q_flat.stride(), *K_flat.stride(), *V_flat.stride(),
            *O_flat.stride(), *L_flat.stride(),
            N_QUERIES = N_Q, N_KEYS = N_K,
            scale = scale,
            D = d, Q_TILE_SIZE = QTS, K_TILE_SIZE = KTS,  # pyright: ignore[reportArgumentType]
            is_causal=is_causal    # pyright: ignore[reportArgumentType]
        )

        # unflatten O and L
        Output = O_flat.view(*Q.shape)
        LSE = L_flat.view(*Q.shape[:-1])

        ctx.save_for_backward(Q, K, V, Output, LSE)
        ctx.is_causal = is_causal

        return Output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
