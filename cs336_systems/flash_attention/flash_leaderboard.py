# ruff: noqa: F841, E741
# pyright: reportUnreachable=false, reportOptionalMemberAccess=none
import math

import torch
import triton
from jaxtyping import Float
from torch.types import Tensor
from triton import language as tl

# model config
N_HEADS = 16
SEQ_LEN = 16384
HEAD_DIM = 64

SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# using exp2/log2: exp(x)=2^(x/ln2)
LN2 = 0.6931471824645996
RCP_LN2 = 1.4426950408889634
QK_SCALE_LOG2 = SM_SCALE * RCP_LN2

# Forward Tile Sizes
FWD_QTS = 128
FWD_KTS = 64

# Backward Tile Sizes
BWD_QTS = 64
BWD_KTS = 128
BWD_DTS = 128

# tiles
T_q = (SEQ_LEN + FWD_QTS - 1) // FWD_QTS
T_k = (SEQ_LEN + FWD_KTS - 1) // FWD_KTS


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
    q_start = query_tile_index * Q_TILE_SIZE
    q_indices = q_start + tl.arange(0, Q_TILE_SIZE)

    # make_block_ptr: returns a pointer to a block in a parent tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # launch grid (batch_size, q_tiles)
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # in a batch, find tiles for this thread
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_T_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),  # 1. 逻辑形状互换：从 (N, D) 变为 (D, N)
        strides=(stride_kd, stride_kk),  # 2. 步长互换：原本跨行现在跨列，原本跨列现在跨行
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),  # 3. 块形状互换：确保计算维度 (D) 在前
        order=(0, 1),  # 4. 存储顺序：告诉硬件物理上 D 是连续的还是 N 是连续的
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
    q = tl.load(Q_block_ptr)  # bf16

    # Separate the non-masked tiles from the tile diagonals
    # mask 是一个下三角矩阵，对于一个固定的需要处理的 Q_tile，所有的 K(i) 被切成两部分
    # 前面的是 Q 能看到的，中间的被对角线穿过需要 mask，后面的看不到全都跳过

    # --- 阶段 1: 完全不需要掩码的循环 ---
    # 所有 (i+1)*KTS <= q_start 的 K 块（q_start 之前可以看到的内容）都属于此范围
    num_safe_steps = q_start // K_TILE_SIZE
    for i in tl.range(0, num_safe_steps):
        # 确保整除，在 A100 移除 boundary_check，H100 有 TMA 可忽略
        k = tl.load(K_T_block_ptr)
        v = tl.load(V_block_ptr)

        # Tensor Core only accept 16b operand
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
    # 覆盖从 q_start 到 q_start + QTS 范围内的 K，也就是被对角线切分的 K 块
    # 起始位置是 num_safe_steps * KTS，前面的都被阶段一处理完了，后面的跳过
    start_step = num_safe_steps
    # 结束位置是 (q_start + QTS) / KTS
    end_step = tl.minimum(tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE), tl.cdiv(N_KEYS, K_TILE_SIZE))

    for i in tl.range(start_step, end_step):
        k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        k = tl.load(K_T_block_ptr)
        v = tl.load(V_block_ptr)

        s = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)) * scale

        # 仅在此阶段应用 Causal Mask
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
    l = m + tl.log2(l)

    # save to HBM
    tl.store(O_block_ptr, o.to(q.dtype))
    tl.store(L_block_ptr, l)


@triton.jit
def flash_bwd_D_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_Db, stride_Dq,
    N_QUERIES,
    d_model: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    pid_D = tl.program_id(0)  # tile index of D
    pid_bh = tl.program_id(1)  # index of batch * header

    # init block_ptr of O and dO
    o_block_ptr = tl.make_block_ptr(
        base=O_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, d_model),
        strides=(stride_oq, stride_od),
        offsets=(pid_D * D_TILE_SIZE, 0),
        block_shape=(D_TILE_SIZE, d_model),
        order=(1, 0),  # contiguous on row(d_model)
    )

    do_block_ptr = tl.make_block_ptr(
        base=dO_ptr + pid_bh * stride_ob,
        shape=(N_QUERIES, d_model),
        strides=(stride_oq, stride_od),
        offsets=(pid_D * D_TILE_SIZE, 0),
        block_shape=(D_TILE_SIZE, d_model),
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
    o = tl.load(o_block_ptr, boundary_check=(0,)).to(tl.float32)
    do = tl.load(do_block_ptr, boundary_check=(0,)).to(tl.float32)

    # D = rowsum(o * do)
    D = tl.sum(o * do, axis=-1)

    tl.store(D_block_ptr, D.to(tl.float32), boundary_check=(0,))


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
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_q_base = pid_bh * stride_qb
    off_k_base = pid_bh * stride_kb
    off_v_base = pid_bh * stride_vb
    off_d_base = pid_bh * stride_Db
    off_l_base = pid_bh * stride_lb

    # --- 1. 初始化“驻留”块的指针 (针对当前 pid_q) ---
    # Q, dO, L, D 及其输出目标 dQ 在整个 Kernel 执行期间，其行偏移是固定的
    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + off_q_base,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=dO_ptr + off_q_base,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=L_ptr + off_l_base,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    D_row_block_ptr = tl.make_block_ptr(
        base=D_ptr + off_d_base,
        shape=(N_QUERIES, 1),
        strides=(stride_Dq, 1),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # 加载驻留数据
    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    do = tl.load(do_block_ptr, boundary_check=(0, 1))
    l_i = tl.load(l_block_ptr, boundary_check=(0, 1))
    di = tl.load(D_row_block_ptr, boundary_check=(0, 1))

    # 初始化局部 dQ 寄存器
    dq = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # --- 2. 在循环外初始化“流式”块的起始指针 (针对 K, V) ---
    # K, V 从列 0 开始移动
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + off_k_base,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V_ptr + off_v_base,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 3. 循环遍历 K, V
    curr_n_end = (pid_q + 1) * Q_TILE_SIZE if is_causal else N_KEYS

    for start_n in range(0, curr_n_end, K_TILE_SIZE):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        # --- 计算核心逻辑 ---
        s = tl.dot(q, k.T) * scale
        l_i_scaled = l_i * 0.69314718
        p = tl.exp(s - l_i_scaled)
        if is_causal:
            mask_cond = start_n + K_TILE_SIZE > pid_q * Q_TILE_SIZE
            if mask_cond:
                offs_q = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                offs_k = start_n + tl.arange(0, K_TILE_SIZE)
                p = tl.where(offs_q[:, None] >= offs_k[None, :], p, 0.0)

        dp = tl.dot(do, v.T)
        ds = p * (dp - di) * scale
        dq += tl.dot(ds.to(tl.bfloat16), k)

        # --- 推进流式指针 ---
        # 沿着维度 0 (在 K/V 矩阵中即为序列长度维度) 推进 K_TILE_SIZE
        k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (K_TILE_SIZE, 0))

    # --- 4. 写回结果 ---
    dq_out_ptr = tl.make_block_ptr(
        base=dQ_ptr + off_q_base,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dq_out_ptr, dq.to(tl.bfloat16), boundary_check=(0, 1))


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
    scale,
    D: tl.constexpr,  # Head Dimension
    Q_TILE_SIZE: tl.constexpr,  # BLOCK_M (Query 方向分块)
    K_TILE_SIZE: tl.constexpr,  # BLOCK_N (KV 方向分块)
    is_causal: tl.constexpr,
):
    # 1. 获取程序索引
    pid_k = tl.program_id(0)  # 当前处理第几个 KV 块
    pid_bh = tl.program_id(1)  # 当前处理第几个 Batch/Head

    # 计算 Batch 级别的基础偏移
    off_q_base = pid_bh * stride_qb
    off_k_base = pid_bh * stride_kb
    off_v_base = pid_bh * stride_vb
    off_d_base = pid_bh * stride_Db
    off_l_base = pid_bh * stride_lb

    # --- 2. 初始化驻留 SRAM 的块 (K, V, 以及输出 dK, dV) ---
    # 这些块在整个 pid_k 的生命周期内，行/列偏移是固定的
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + off_k_base,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V_ptr + off_v_base,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 加载驻留数据
    k = tl.load(k_block_ptr, boundary_check=(0, 1))
    v = tl.load(v_block_ptr, boundary_check=(0, 1))

    # 初始化局部梯度累加器
    dk = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dv = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    # --- 3. 初始化流式块的起始指针 (Q, dO, L, D) ---
    # 因果掩码优化：如果当前是第 j 个 KV 块，它只会被行号 i >= j 的 Query 块看到
    # 因此从 start_m = pid_k * K_TILE_SIZE 开始遍历，直接跳过全 0 区域
    start_m_idx = pid_k * K_TILE_SIZE if is_causal else 0

    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + off_q_base,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(start_m_idx, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    do_block_ptr = tl.make_block_ptr(
        base=dO_ptr + off_q_base,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(start_m_idx, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # L 和 D 项视为 (N, 1) 的列向量
    l_block_ptr = tl.make_block_ptr(
        base=L_ptr + off_l_base,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(start_m_idx, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    D_row_block_ptr = tl.make_block_ptr(
        base=D_ptr + off_d_base,
        shape=(N_QUERIES, 1),
        strides=(stride_Dq, 1),
        offsets=(start_m_idx, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # --- 4. 迭代遍历 Query 块 ---
    for start_m in range(start_m_idx, N_QUERIES, Q_TILE_SIZE):
        # 加载流式数据
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        do = tl.load(do_block_ptr, boundary_check=(0, 1))
        l_i = tl.load(l_block_ptr, boundary_check=(0, 1))
        di = tl.load(D_row_block_ptr, boundary_check=(0, 1))

        # 计算注意力分数 S = (Q @ K.T) * scale
        s = tl.dot(q, k.T) * scale

        # 处理对角线块与非对角线块的因果掩码
        if is_causal and (start_m < (pid_k + 1) * K_TILE_SIZE):
            # 对角线块：需要构造 mask 进行过滤
            # 获取当前块内相对于序列的绝对坐标
            offs_m = start_m + tl.arange(0, Q_TILE_SIZE)
            offs_k = pid_k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            p = tl.exp(s - l_i * 0.69314718)
            # 因果限制：Row Index >= Col Index
            p = tl.where(offs_m[:, None] >= offs_k[None, :], p, 0.0)
        else:
            # 非对角线块：已经在 start_m_idx 的过滤下，全是有效计算区域
            p = tl.exp(s - l_i * 0.69314718)

        # -- 计算 dV 贡献 --
        # dV = P.T @ dO
        dv += tl.dot(p.T.to(tl.bfloat16), do)

        # -- 计算 dK 贡献 --
        # 根据 FlashAttention-2 论文公式 [cite: 1230-1232, 2958]
        # dS = P * (dP - D), 其中 dP = dO @ V.T
        dp = tl.dot(do, v.T)
        ds = p * (dp - di) * scale
        # dK = dS.T @ Q
        dk += tl.dot(ds.T.to(tl.bfloat16), q)

        # --- 推进流式指针 ---
        q_block_ptr = tl.advance(q_block_ptr, (Q_TILE_SIZE, 0))
        do_block_ptr = tl.advance(do_block_ptr, (Q_TILE_SIZE, 0))
        l_block_ptr = tl.advance(l_block_ptr, (Q_TILE_SIZE, 0))
        D_row_block_ptr = tl.advance(D_row_block_ptr, (Q_TILE_SIZE, 0))

    # --- 5. 写回结果 (Col-Parallel 无冲突) ---
    dk_out_ptr = tl.make_block_ptr(
        base=dK_ptr + off_k_base,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dv_out_ptr = tl.make_block_ptr(
        base=dV_ptr + off_v_base,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dk_out_ptr, dk.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(dv_out_ptr, dv.to(tl.bfloat16), boundary_check=(0, 1))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... seq_len d_k"],
        K: Float[Tensor, "... seq_len d_k"],
        V: Float[Tensor, "... seq_len d_k"],
        is_causal=False,
    ):
        BH = Q.shape[0]
        # init the return value: Output and LogSumExp, LSE need float32
        O = torch.empty_like(Q, device="cuda", dtype=torch.bfloat16)
        L = torch.empty((BH, SEQ_LEN), device="cuda", dtype=torch.float32)

        flash_fwd_kernel[(T_q, BH)](
            Q, K, V,
            O, L,
            *Q.stride(), *K.stride(), *V.stride(),
            *O.stride(), *L.stride(),
            N_QUERIES = SEQ_LEN, N_KEYS = SEQ_LEN,
            scale = QK_SCALE_LOG2,
            D = HEAD_DIM, Q_TILE_SIZE = FWD_QTS, K_TILE_SIZE = FWD_KTS,  # pyright: ignore[reportArgumentType]
            is_causal=is_causal,    # pyright: ignore[reportArgumentType]
            num_warps=4, num_stages=4,  # pyright: ignore[reportCallIssue]
        )

        # unflatten O and L
        Output = O.view(*Q.shape)
        LSE = L.view(*Q.shape[:-1])

        ctx.save_for_backward(Q, K, V, Output, LSE)
        ctx.is_causal = is_causal

        return Output

    @staticmethod
    def backward(
        ctx,
        grad_out: Float[Tensor, "... N_q d"],
    ):
        Q: torch.Tensor
        K: torch.Tensor
        V: torch.Tensor
        Output: torch.Tensor
        LSE: torch.Tensor

        Q, K, V, Output, LSE = ctx.saved_tensors
        Q_flat = Q.view(-1, Q.shape[-2], Q.shape[-1]).contiguous()
        K_flat = K.view(-1, K.shape[-2], K.shape[-1]).contiguous()
        V_flat = V.view(-1, V.shape[-2], V.shape[-1]).contiguous()
        Output_flat = Output.view(-1, Output.shape[-2], Output.shape[-1]).contiguous()
        dO_flat = grad_out.view(-1, grad_out.shape[-2], grad_out.shape[-1]).contiguous()
        LSE_flat = LSE.view(-1, LSE.shape[-1]).contiguous()

        BH = Q_flat.shape[0]
        N_Q = Q_flat.shape[1]
        N_K = K_flat.shape[1]

        # init dQ/dK/dV/Delta
        dQ_flat = torch.empty_like(Q_flat)
        dK_flat = torch.empty_like(K_flat)
        dV_flat = torch.empty_like(V_flat)
        D = torch.empty_like(LSE_flat, device=Q.device, dtype=torch.float32)

        # compute D = rowsum(dO ◦ O)
        flash_bwd_D_kernel[(triton.cdiv(N_Q, BWD_DTS), BH)](
            Output_flat,
            dO_flat,
            D,
            *dO_flat.stride(),
            *D.stride(),
            N_QUERIES=N_Q,
            d_model=HEAD_DIM,
            D_TILE_SIZE=BWD_DTS,  # pyright: ignore[reportArgumentType]
            num_warps=4,  # pyright: ignore[reportCallIssue]
        )

        flash_bwd_dq_kernel[(triton.cdiv(N_Q, BWD_QTS), BH)](
            Q_flat,
            K_flat,
            V_flat,
            LSE_flat,
            dO_flat,
            dQ_flat,
            D,
            *Q_flat.stride(),
            *K_flat.stride(),
            *V_flat.stride(),
            *Output_flat.stride(),
            *LSE_flat.stride(),
            *D.stride(),
            N_QUERIES=N_Q,
            N_KEYS=N_K,
            scale=SM_SCALE,
            D=HEAD_DIM,
            Q_TILE_SIZE=BWD_QTS,
            K_TILE_SIZE=BWD_KTS,  # pyright: ignore[reportArgumentType]
            is_causal=ctx.is_causal,  # pyright: ignore[reportArgumentType]
            num_warps=8,
            num_stages=4,
        )

        flash_bwd_dkdv_kernel[(triton.cdiv(N_K, BWD_KTS), BH)](
            Q_flat,
            K_flat,
            V_flat,
            LSE_flat,
            dO_flat,
            dK_flat,
            dV_flat,
            D,
            *Q_flat.stride(),
            *K_flat.stride(),
            *V_flat.stride(),
            *Output_flat.stride(),
            *LSE_flat.stride(),
            *D.stride(),
            N_QUERIES=N_Q,
            N_KEYS=N_K,
            scale=SM_SCALE,
            D=HEAD_DIM,
            Q_TILE_SIZE=BWD_QTS,
            K_TILE_SIZE=BWD_KTS,  # pyright: ignore[reportArgumentType]
            is_causal=ctx.is_causal,  # pyright: ignore[reportArgumentType]
            num_warps=8,
            num_stages=4,
        )

        # unflatten O and L
        dQ = dQ_flat.view(Q.shape)
        dK = dK_flat.view(K.shape)
        dV = dV_flat.view(V.shape)

        return dQ, dK, dV, None


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    flash = FlashAttention2Triton.apply

    def flash_forward_backward():
        # clear grad to avoid accum
        q.grad = None
        k.grad = None
        v.grad = None

        o = flash(q, k, v, True)
        o.sum().backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=100)
    print(results)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    test_timing_flash_forward_backward()
