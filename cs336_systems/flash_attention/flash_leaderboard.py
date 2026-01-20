# ruff: noqa: F841, E741
# pyright: reportUnreachable=false, reportOptionalMemberAccess=none, reportIndexIssue=none
import math

import torch
import triton
from einops import rearrange
from jaxtyping import Float
from torch.types import Tensor
from triton import language as tl

# using exp2/log2: exp(x)=2^(x/ln2)
HEAD_DIM = 64
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
LN2 = 0.6931471824645996
RCP_LN2 = 1.4426950408889634
QK_SCALE_LOG2 = SM_SCALE * RCP_LN2

# Forward Tile Sizes
FWD_QTS = 128
FWD_KTS = 64

# Backward Tile Sizes
BWD_QTS = 128
BWD_KTS = 32
BWD_DTS = 32


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,  # stride of: batch, seq_len, d_model
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale, scale_back,
    D_MODEL: tl.constexpr,  # d_model/d_head: dim of row
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # row indices for the current Q tile
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    # make_block_ptr: returns a pointer to a block in a parent tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # launch grid (batch_size, q_tiles)
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # in a batch, find tiles for this thread
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    # transpose K by swapping shape and strides
    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D_MODEL, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D_MODEL, K_TILE_SIZE),
        order=(0, 1),  # K is row-major, transpose K is column-major
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
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
    o = tl.zeros((Q_TILE_SIZE, D_MODEL), tl.float32)  # register file(RF) is larger/faster than SRAM

    # each thread run with a tile of Q, load: HBM -> SRAM(Shared Mem) -> Register File(RF)
    q = tl.load(Q_block_ptr)  # bf16

    # Separate the non-masked tiles from the tile diagonals
    # mask 是一个下三角矩阵，对于一个固定的需要处理的 Q_tile，所有的 K(i) 被切成两部分
    # 前面的是 Q 能看到的，中间的被对角线穿过需要 mask，后面的看不到全都跳过

    # --- 阶段 1: 完全不需要掩码的循环 ---
    # 所有 (i+1)*KTS <= q_start 的 K 块（q_start 之前可以看到的内容）或者不需要 causal mask 时遍历整个 K
    num_safe_steps = tl.minimum(q_start // K_TILE_SIZE, N_KEYS // K_TILE_SIZE)
    num_safe_steps = num_safe_steps if is_causal else N_KEYS // K_TILE_SIZE
    for i in tl.range(0, num_safe_steps):
        # 在 A100 确保整除，移除 boundary_check，H100 有 TMA 可忽略
        k = tl.load(K_T_block_ptr)
        v = tl.load(V_block_ptr)

        # Compute tile of pre-softmax attention scores
        s = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)) * scale

        # Online Softmax
        _m = tl.maximum(m, tl.max(s, axis=-1))
        p = tl.exp2(s - _m[:, None])
        rescale = tl.exp2(m - _m)
        l = rescale * l + tl.sum(p, axis=-1)
        o = o * rescale[:, None]
        o = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=o)
        m = _m

        K_T_block_ptr = tl.advance(K_T_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # --- 阶段 2: 需要对角线掩码的循环 ---
    if is_causal:
        # 结束位置是 (q_start + QTS) / KTS，或者非因果注意力时，处理 K 的未对齐边界
        end_step = tl.minimum(tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE), tl.cdiv(N_KEYS, K_TILE_SIZE))

        # 覆盖从 q_start 到 q_start + QTS 范围内的 K，也就是被对角线切分的 K 块
        # 起始位置是 num_safe_steps * KTS，前面的都被阶段一处理完了，后面的跳过
        for i in tl.range(num_safe_steps, end_step):
            k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            k = tl.load(K_T_block_ptr)
            v = tl.load(V_block_ptr)

            s = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)) * scale

            # 确保不越过 N_KEYS，叠加 Causal Mask
            mask = q_indices[:, None] >= k_indices[None, :]
            s = tl.where(mask, s, float("-inf"))

            # Softmax 更新逻辑同上...
            _m = tl.maximum(m, tl.max(s, axis=-1))
            p = tl.exp2(s - _m[:, None])
            rescale = tl.exp2(m - _m)
            l = rescale * l + tl.sum(p, axis=-1)
            o = o * rescale[:, None]
            o = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=o)
            m = _m

            K_T_block_ptr = tl.advance(K_T_block_ptr, (0, K_TILE_SIZE))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # compute the final output and lse
    o = o * (1.0 / l)[:, None]
    l = (m + tl.log2(l)) * scale_back

    # save to HBM
    tl.store(O_block_ptr, o.to(q.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))


@triton.jit
def flash_bwd_D_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_Db, stride_Dq,
    N_QUERIES: tl.constexpr,
    D_MODEL: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    pid_D = tl.program_id(0)  # tile index of D
    pid_bh = tl.program_id(1)  # index of batch * header

    # init block_ptr of O and dO
    o_block_ptr = tl.make_block_ptr(
        base=O_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(pid_D * D_TILE_SIZE, 0),
        block_shape=(D_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    do_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(pid_D * D_TILE_SIZE, 0),
        block_shape=(D_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + pid_bh * stride_Db,
        shape=(N_QUERIES,),
        strides=(stride_Dq,),
        offsets=(pid_D * D_TILE_SIZE,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # boundary_check=(0,) check seq_len dim
    o = tl.load(o_block_ptr).to(tl.float32)
    do = tl.load(do_block_ptr).to(tl.float32)

    # D = rowsum(o * do)
    D = tl.sum(o * do, axis=-1)

    tl.store(D_block_ptr, D.to(tl.float32))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
    dQ_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    N_QUERIES, N_KEYS,
    scale: tl.constexpr,
    scale_to_log2: tl.constexpr,
    sm_scale: tl.constexpr,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    dtype = dQ_ptr.dtype.element_ty

    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_bh * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=L_ptr + pid_bh * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + pid_bh * stride_Db,
        shape=(N_QUERIES, 1),
        strides=(stride_Dq, 1),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # 加载 K^T 和 V^T 方便直接 dot 计算
    kT_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_bh * stride_kb,
        shape=(D_MODEL, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D_MODEL, K_TILE_SIZE),
        order=(0, 1),
    )
    vT_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_bh * stride_vb,
        shape=(D_MODEL, N_KEYS),
        strides=(stride_vd, stride_vk),
        offsets=(0, 0),
        block_shape=(D_MODEL, K_TILE_SIZE),
        order=(0, 1),
    )
    # not transpose K for: dQ = dS @ K
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_bh * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    # load to SRAM/RF
    q = tl.load(q_block_ptr)
    do = tl.load(dO_block_ptr)
    l_i = tl.load(l_block_ptr)
    d_i = tl.load(D_block_ptr)
    dq = tl.zeros([Q_TILE_SIZE, D_MODEL], dtype=tl.float32)

    # 3. 循环遍历 K, V
    q_start = pid_q * Q_TILE_SIZE

    # --- 阶段 1 无 Mask：因果模式下只走到对角线前；非因果下走到尽头 ---
    num_safe_steps = q_start // K_TILE_SIZE if is_causal else (N_KEYS // K_TILE_SIZE)
    for i in tl.range(0, num_safe_steps):
        kT = tl.load(kT_block_ptr)
        vT = tl.load(vT_block_ptr)
        k = tl.load(k_block_ptr)

        # S = (Q @ K^T) * scale (注：此处 scale 应为 QK_SCALE_LOG2)
        s = tl.dot(q, kT) * scale
        # P = exp(S * ln2 - LSE)  将 base-2 的 S 转回自然对数域与 LSE 抵消
        p = tl.exp2(s - l_i * scale_to_log2)

        # dP = dO @ V^T
        dp = tl.dot(do.to(tl.bfloat16), vT.to(tl.bfloat16))
        # dS = P * (dP - D) * scale，这里的 QK_SCALE_LOG2 与后面的 dQ * scale_back 合并为 * sm_scale
        ds = p * (dp - d_i)

        # dQ += dS @ K
        dq = tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16), acc=dq)

        kT_block_ptr = tl.advance(kT_block_ptr, (0, K_TILE_SIZE))
        vT_block_ptr = tl.advance(vT_block_ptr, (0, K_TILE_SIZE))
        k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))

    # --- 阶段 2: 需要 Causal Mask 的对角线块 ---
    if is_causal:
        end_step = tl.minimum(tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE), tl.cdiv(N_KEYS, K_TILE_SIZE))
        q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

        for i in tl.range(num_safe_steps, end_step):
            k_start = i * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)

            kT = tl.load(kT_block_ptr)
            vT = tl.load(vT_block_ptr)
            k = tl.load(k_block_ptr)

            s = tl.dot(q.to(tl.bfloat16), kT.to(tl.bfloat16)) * scale
            p = tl.exp2(s - l_i * scale_to_log2)

            # 应用 Causal Mask
            mask = q_indices[:, None] >= k_indices[None, :]
            p = tl.where(mask, p, 0.0)

            dp = tl.dot(do.to(tl.bfloat16), vT.to(tl.bfloat16))
            ds = p * (dp - d_i)
            dq = tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16), acc=dq)

            kT_block_ptr = tl.advance(kT_block_ptr, (0, K_TILE_SIZE))
            vT_block_ptr = tl.advance(vT_block_ptr, (0, K_TILE_SIZE))
            k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))

    dQ_ptr = tl.make_block_ptr(
        base=dQ_ptr + pid_bh * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    tl.store(dQ_ptr, (dq * sm_scale).to(dtype))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    L_ptr, D_ptr,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    scale: tl.constexpr,
    scale_to_log2: tl.constexpr,
    sm_scale: tl.constexpr,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    MASK_SLICE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_kv = tl.program_id(0)  # kv-block id
    pid_bh = tl.program_id(1)  # batch-size id

    base_q = pid_bh * N_Q * D_MODEL
    base_k = pid_bh * N_K * D_MODEL
    base_l = pid_bh * N_Q

    start_kv = pid_kv * K_TILE_SIZE
    offs_kv = start_kv + tl.arange(0, K_TILE_SIZE)
    mask_kv = offs_kv < N_K
    offs_d = tl.arange(0, D_MODEL)

    # load K and V, K is scaled to exp2/log2 here
    k = tl.load(K_ptr + base_k + offs_kv[:, None] * D_MODEL + offs_d[None, :], mask=mask_kv[:, None], other=0.0) * scale  # [B, D]
    v = tl.load(V_ptr + base_k + offs_kv[:, None] * D_MODEL + offs_d[None, :], mask=mask_kv[:, None], other=0.0)  # [B, D]

    dK = tl.zeros((K_TILE_SIZE, D_MODEL), tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D_MODEL), tl.float32)

    # causal：对角区域切小（mask 只在必要块做）
    # 对角区域：q in [start_n, start_n+BLOCK_N)
    # 非对角：q in [start_n+BLOCK_N, N_CTX)
    if is_causal:
        num_diag_steps = K_TILE_SIZE // MASK_SLICE
        start_q = start_kv
        for _ in range(num_diag_steps):
            offs_q = start_q + tl.arange(0, MASK_SLICE)
            mask_q = offs_q < N_Q

            qT = tl.load( Q_ptr + base_q + offs_q[None, :] * D_MODEL + offs_d[:, None], mask=mask_q[None, :], other=0.0)  # [D, B]
            dO = tl.load(dO_ptr + base_q + offs_q[:, None] * D_MODEL + offs_d[None, :], mask=mask_q[:, None], other=0.0)  # [B, D]
            l = tl.load(L_ptr + base_l + offs_q, mask=mask_q, other=0.0) * scale_to_log2  # [B]
            D = tl.load(D_ptr + base_l + offs_q, mask=mask_q, other=0.0)  # [B]

            qkT = tl.dot(k, qT)  # [BN, BM]
            # for pass test save L to exp/log domain, now scale to exp2/log2 domain
            pT = tl.exp2(qkT - l[None, :])  # [BN, BM]

            # causal mask for this diagonal slice
            mask = offs_q[None, :] >= offs_kv[:, None]
            pT = tl.where(mask & mask_q[None, :] & mask_kv[:, None], pT, 0.0)

            dV = tl.dot(pT.to(tl.bfloat16), dO.to(tl.bfloat16), acc=dV)

            dpT = tl.dot(v, tl.trans(dO)).to(tl.float32)  # [BN, BM]
            dsT = pT * (dpT - D[None, :])  # dL/ds2

            dK = tl.dot(dsT.to(tl.bfloat16), tl.trans(qT).to(tl.bfloat16), acc=dK)
            start_q += MASK_SLICE

        start_q = start_kv + num_diag_steps * MASK_SLICE
    else:
        start_q = 0

    # non-masked part (q >= start_m)
    num_total_steps = tl.cdiv(N_Q - start_q, Q_TILE_SIZE)
    for step in range(num_total_steps):
        offs_q = start_q + step * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        mask_q = offs_q < N_Q

        qT = tl.load(Q_ptr + base_q + offs_q[None, :] * D_MODEL + offs_d[:, None], mask=mask_q[None, :], other=0.0)
        dO = tl.load(dO_ptr + base_q + offs_q[:, None] * D_MODEL + offs_d[None, :], mask=mask_q[:, None], other=0.0)
        l = tl.load(L_ptr + base_l + offs_q, mask=mask_q, other=0.0) * scale_to_log2
        D = tl.load(D_ptr + base_l + offs_q, mask=mask_q, other=0.0)

        qkT = tl.dot(k, qT)
        pT = tl.exp2(qkT - l[None, :])
        # Mask out-of-bounds Q for correctness
        pT = tl.where(mask_q[None, :] & mask_kv[:, None], pT, 0.0)

        dV = tl.dot(pT.to(tl.bfloat16), dO.to(tl.bfloat16), acc=dV)

        dpT = tl.dot(v, tl.trans(dO)).to(tl.float32)
        dsT = pT * (dpT - D[None, :])

        dK = tl.dot(dsT.to(tl.bfloat16), tl.trans(qT).to(tl.bfloat16), acc=dK)

    # write back
    tl.store(dV_ptr + base_k + offs_kv[:, None] * D_MODEL + offs_d[None, :], dV.to(tl.bfloat16), mask=mask_kv[:, None])
    tl.store(dK_ptr + base_k + offs_kv[:, None] * D_MODEL + offs_d[None, :], (dK * sm_scale).to(tl.bfloat16), mask=mask_kv[:, None],)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... seq_len d"],
        K: Float[Tensor, "... seq_len d"],
        V: Float[Tensor, "... seq_len d"],
        is_causal=False,
    ):
        # record the original shape, compatible 3d/4d/5d
        *B, N_Q, d = Q.shape
        N_K = K.shape[-2]
        device = Q.device
        dtype = Q.dtype

        # check QKV is in cuda
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert N_Q % FWD_QTS == 0, "Q sequence length must be multiple of FWD_QTS"
        assert N_K % FWD_KTS == 0, "K sequence length must be multiple of FWD_KTS"

        # flatten Q/K/V for apply Algorithm 1
        Q_flat = rearrange(Q, "... N d -> (...) N d").contiguous()  # (merged_B seq_len d_k)
        K_flat = rearrange(K, "... N d -> (...) N d").contiguous()
        V_flat = rearrange(V, "... N d -> (...) N d").contiguous()
        m_B = Q_flat.shape[0]  # get merged B

        # tile count of Q, K/V
        T_q = (N_Q + FWD_QTS - 1) // FWD_QTS

        # init the return value: Output and LogSumExp, need float32 for LSE
        O_flat = torch.empty_like(Q_flat, device=device, dtype=dtype)
        L_flat = torch.empty((m_B, N_Q), device=device, dtype=torch.float32)

        flash_fwd_kernel[(T_q, m_B)](
            Q_flat, K_flat, V_flat,
            O_flat, L_flat,
            *Q_flat.stride(), *K_flat.stride(), *V_flat.stride(),
            *O_flat.stride(), *L_flat.stride(),
            N_QUERIES = N_Q, N_KEYS = N_K,
            scale = QK_SCALE_LOG2, scale_back = LN2,
            D_MODEL = d, Q_TILE_SIZE = FWD_QTS, K_TILE_SIZE = FWD_KTS,  # pyright: ignore[reportArgumentType]
            is_causal=is_causal,
            num_warps=4, num_stages=4,  # pyright: ignore[reportCallIssue]
        )

        ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
        ctx.is_causal = is_causal
        ctx.QKV_shape = (Q.shape, K.shape, V.shape)

        # unflatten O and L
        Output = O_flat.view(Q.shape)
        return Output

    @staticmethod
    def backward(
        ctx,
        grad_out: Float[Tensor, "... N_q d"],
    ):
        Q_flat: torch.Tensor
        K_flat: torch.Tensor
        V_flat: torch.Tensor
        O_flat: torch.Tensor
        L_flat: torch.Tensor

        Q_flat, K_flat, V_flat, O_flat, L_flat = ctx.saved_tensors
        device = Q_flat.device

        B, N_Q, d = Q_flat.shape
        N_K = K_flat.shape[-2]

        dO_flat = rearrange(grad_out, "... N d -> (...) N d").contiguous()

        # tile count of Q, K/V
        T_q = triton.cdiv(N_Q, BWD_QTS)
        T_k = triton.cdiv(N_K, BWD_KTS)

        # compute D = rowsum(dO ◦ O)
        D = torch.empty_like(L_flat, device=device, dtype=torch.float32)
        flash_bwd_D_kernel[(triton.cdiv(N_Q, BWD_DTS), B)](
            O_flat, dO_flat, D,
            *dO_flat.stride(), *D.stride(),
            N_QUERIES=N_Q, D_MODEL=d, D_TILE_SIZE=BWD_DTS,
            num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
        )

        # init dQ/dK/dV
        dQ_flat = torch.empty_like(Q_flat)
        dK_flat = torch.empty_like(K_flat)
        dV_flat = torch.empty_like(V_flat)

        flash_bwd_dq_kernel[(T_q, B)](
            Q_flat, K_flat, V_flat, L_flat,
            dO_flat, dQ_flat, D,
            *Q_flat.stride(), *K_flat.stride(), *V_flat.stride(),
            *O_flat.stride(), *L_flat.stride(), *D.stride(),
            N_QUERIES=N_Q, N_KEYS=N_K,
            scale=QK_SCALE_LOG2, scale_to_log2=RCP_LN2, sm_scale=SM_SCALE,
            D_MODEL=d, Q_TILE_SIZE=BWD_QTS, K_TILE_SIZE=BWD_KTS,
            is_causal=ctx.is_causal,
            num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
        )

        flash_bwd_dkdv_kernel[(T_k, B)](
            Q_flat, K_flat, V_flat,
            dO_flat, dK_flat, dV_flat, L_flat, D,
            N_Q=N_Q, N_K=N_K,
            scale=QK_SCALE_LOG2, scale_to_log2=RCP_LN2, sm_scale=SM_SCALE,
            D_MODEL=d, Q_TILE_SIZE=BWD_QTS, K_TILE_SIZE=BWD_KTS,
            MASK_SLICE=BWD_KTS // 2,
            is_causal=ctx.is_causal,
            num_warps=4, num_stages=2,  # pyright: ignore[reportCallIssue]
        )

        # unflatten
        dQ = dQ_flat.view(ctx.QKV_shape[0])
        dK = dK_flat.view(ctx.QKV_shape[1])
        dV = dV_flat.view(ctx.QKV_shape[2])

        return dQ, dK, dV, None


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device="cuda",
        dtype=torch.bfloat16, requires_grad=True
    )

    flash = FlashAttention2Triton.apply

    def flash_forward_backward():
        # clear grad to avoid accum
        q.grad = None
        k.grad = None
        v.grad = None

        o = flash(q, k, v, True)
        o.sum().backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    test_timing_flash_forward_backward()
