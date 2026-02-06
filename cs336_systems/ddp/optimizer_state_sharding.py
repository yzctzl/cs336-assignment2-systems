# pyright: reportAttributeAccessIssue=none
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[Any], optimizer_cls: type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 核心状态维护
        self.rank_loads = [0] * self.world_size
        self.sub_optimizer: Optimizer = None
        self.param_to_rank: dict[torch.nn.Parameter, int] = {}

        # 为了高效通信，按 Rank 存储参数引用
        self.params_by_rank: list[list[torch.nn.Parameter]] = [[] for _ in range(self.world_size)]

        # 调用父类初始化，这会触发 add_param_group
        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        基于 numel 贪心算法的分片逻辑。
        确保所有 Rank 算出的分配方案完全一致。
        """
        params = list(param_group["params"])
        # 备份原始参数列表，用于后续通信
        original_params = params

        # 1. 负载均衡分配
        sharded_params_for_cur_rank = []
        for p in original_params:
            p: torch.nn.Parameter
            if not p.requires_grad:
                continue

            # 找到当前载荷最小的 Rank (Greedy)
            min_load_rank = self.rank_loads.index(min(self.rank_loads))

            # 记录归属关系
            self.param_to_rank[p] = min_load_rank
            self.params_by_rank[min_load_rank].append(p)
            self.rank_loads[min_load_rank] += p.numel()

            if min_load_rank == self.rank:
                sharded_params_for_cur_rank.append(p)

        # 2. 构造本地子优化器
        # 子优化器只负责更新分配给当前 Rank 的参数组
        local_group = {**param_group, "params": sharded_params_for_cur_rank}

        if self.sub_optimizer is None:
            self.sub_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self.sub_optimizer.add_param_group(local_group)

        # 3. 同步父类状态
        # 注意：父类 param_groups 只维护当前 Rank 负责的参数片，符合 ZeRO-1 理念
        super().add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        """
        1. 备份更新前的参数（用于计算增量或直接覆盖）。
        2. 本地优化器更新分片。
        3. 批量通信同步所有参数。
        """
        # 执行局部更新
        loss = self.sub_optimizer.step(closure, **kwargs)

        # 全局同步更新后的结果
        self._sync_all_params_efficiently()

        return loss

    @torch.no_grad()
    def _sync_all_params_efficiently(self):
        """
        低峰值内存同步：逐参数广播。
        相比扁平化后一次性广播，逐参数广播几乎不产生额外的大缓冲区，
        可以显著降低峰值显存。
        """
        for src_rank in range(self.world_size):
            rank_params = self.params_by_rank[src_rank]
            if not rank_params:
                continue
            for p in rank_params:
                dist.broadcast(p.data, src=src_rank)

    def __getattr__(self, name):
        """转发属性访问给子优化器（如 state 访问）"""
        return getattr(self.sub_optimizer, name)
