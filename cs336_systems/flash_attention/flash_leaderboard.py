# ruff: noqa: F841, E741
# pyright: reportUnreachable=false, reportOptionalMemberAccess=none
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
    scale, scale_to_log2, sm_scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

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
    tl.store(dQ_ptr, (dq * sm_scale).to(q.dtype))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
    dK_ptr, dV_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    N_QUERIES, N_KEYS,
    scale, scale_to_log2, sm_scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)

    kT_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_bh * stride_kb,
        shape=(D_MODEL, N_KEYS),           # 交换维度
        strides=(stride_kd, stride_kk),    # 交换步长
        offsets=(0, pid_k * K_TILE_SIZE),  # 交换偏移 (D从0开始, N从pid_k开始)
        block_shape=(D_MODEL, K_TILE_SIZE),# 加载转置块
        order=(0, 1),                      # D维(第0维)是连续的，所以是 (0, 1)
    )

    vT_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_bh * stride_vb,
        shape=(D_MODEL, N_KEYS),
        strides=(stride_vd, stride_vk),
        offsets=(0, pid_k * K_TILE_SIZE),
        block_shape=(D_MODEL, K_TILE_SIZE),
        order=(0, 1),
    )

    # 初始化起始位置
    _start = pid_k * K_TILE_SIZE if is_causal else 0
    # 总步数：剩余需要处理的 Q 块数量
    num_total_steps = tl.cdiv(tl.maximum(0, N_QUERIES - _start), Q_TILE_SIZE)

    # Mask 步数：只有在对角线重叠区需要 Mask
    # 重叠长度就是 K_TILE_SIZE。例如 K=64, Q=128，重叠 1 个块。
    if is_causal:
        num_masked_steps = tl.cdiv(K_TILE_SIZE, Q_TILE_SIZE)
        # 考虑 N 太小导致 masked_steps > total_steps 的 edge case
        num_masked_steps = tl.minimum(num_masked_steps, num_total_steps)
    else:
        num_masked_steps = 0

    # 无需 mask 步数
    num_safe_steps = num_total_steps - num_masked_steps

    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_bh * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(_start, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    # L, D 加载为列向量
    l_block_ptr = tl.make_block_ptr(
        base=L_ptr + pid_bh * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(_start, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + pid_bh * stride_Db,
        shape=(N_QUERIES, 1),
        strides=(stride_Dq, 1),
        offsets=(_start, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    kT = tl.load(kT_block_ptr)
    vT = tl.load(vT_block_ptr)

    dk = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)
    dv = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)

    _curr = _start
    offs_k = pid_k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    # --- 阶段 1: 对角线块 (Causal) 需要处理 mask ---
    for i in tl.range(0, num_masked_steps):
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        do = tl.load(dO_block_ptr, boundary_check=(0, 1))
        l_i = tl.load(l_block_ptr, boundary_check=(0, 1))
        d_i = tl.load(D_block_ptr, boundary_check=(0, 1))

        s = tl.dot(q.to(tl.bfloat16), kT.to(tl.bfloat16)) * scale
        p = tl.exp2(s - l_i * scale_to_log2)

        offs_q = _curr + tl.arange(0, Q_TILE_SIZE)
        mask = offs_q[:, None] >= offs_k[None, :]
        p = tl.where(mask, p, 0.0)

        # dV += P^T @ dO, Shape: [K_TILE, Q_TILE] @ [Q_TILE, D] -> [K_TILE, D]
        dv += tl.dot(tl.trans(p).to(tl.bfloat16), do.to(tl.bfloat16))

        # dP = dO @ V^T, Shape: [Q_TILE, D] @ [D, K_TILE] -> [Q_TILE, K_TILE]
        dp = tl.dot(do.to(tl.bfloat16), vT.to(tl.bfloat16))
        ds = p * (dp - d_i)

        # dK += dS^T @ Q, Shape: [K_TILE, Q_TILE] @ [Q_TILE, D] -> [K_TILE, D]
        dk += tl.dot(tl.trans(ds).to(tl.bfloat16), q.to(tl.bfloat16))

        # Advance
        q_block_ptr = tl.advance(q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        l_block_ptr = tl.advance(l_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE, 0))
        _curr += Q_TILE_SIZE

    # --- 阶段 2: 无 Mask ---
    for m in tl.range(0, num_safe_steps):
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        do = tl.load(dO_block_ptr, boundary_check=(0, 1))
        l_i = tl.load(l_block_ptr, boundary_check=(0, 1))
        d_i = tl.load(D_block_ptr, boundary_check=(0, 1))

        s = tl.dot(q.to(tl.bfloat16), kT.to(tl.bfloat16)) * scale
        p = tl.exp2(s - l_i * scale_to_log2)

        dv += tl.dot(tl.trans(p).to(tl.bfloat16), do.to(tl.bfloat16))
        dp = tl.dot(do.to(tl.bfloat16), vT.to(tl.bfloat16))
        ds = p * (dp - d_i)
        dk += tl.dot(tl.trans(ds).to(tl.bfloat16), q.to(tl.bfloat16))

        q_block_ptr = tl.advance(q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        l_block_ptr = tl.advance(l_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE, 0))

    # 写回 dK, dV (正常形状，统一缩放)
    dK_out_ptr = tl.make_block_ptr(
        base=dK_ptr + pid_bh * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dV_out_ptr = tl.make_block_ptr(
        base=dV_ptr + pid_bh * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    tl.store(dK_out_ptr, (dk * sm_scale).to(kT.dtype), boundary_check=(0, 1))
    tl.store(dV_out_ptr, dv.to(kT.dtype), boundary_check=(0, 1))


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
            Q_flat, K_flat, V_flat, L_flat,
            dO_flat, dK_flat, dV_flat, D,
            *Q_flat.stride(), *K_flat.stride(), *V_flat.stride(),
            *O_flat.stride(), *L_flat.stride(), *D.stride(),
            N_QUERIES=N_Q, N_KEYS=N_K,
            scale=QK_SCALE_LOG2, scale_to_log2=RCP_LN2, sm_scale=SM_SCALE,
            D_MODEL=d, Q_TILE_SIZE=BWD_QTS, K_TILE_SIZE=BWD_KTS,
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
