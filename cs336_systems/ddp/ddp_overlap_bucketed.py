# pyright: reportAttributeAccessIssue=none, reportOptionalMemberAccess=none
import torch
import torch.distributed as dist
import torch.nn as nn


class DDPOverLapBucket(torch.nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024

        for p in self.module.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)

        self.handles = []
        self.current_bucket = []
        self.current_size = 0

        self._register_hooks()

    def _register_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: nn.Parameter):
        def hook(*unused):
            self.current_bucket.append(param)
            self.current_size += param.grad.numel() * param.grad.element_size()

            if self.current_size >= self.bucket_size_bytes:
                self._dispatch_bucket()

        return hook

    def _dispatch_bucket(self):
        if not self.current_bucket:
            return

        # flatten tensors in current bucket
        grads = [p.grad for p in self.current_bucket]
        flat_grad = torch._utils._flatten_dense_tensors(grads)

        # All Reduce
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG, async_op=True)

        # save for finish_gradient_synchronization
        self.handles.append((handle, flat_grad, grads))

        # reset current bucket
        self.current_bucket = []
        self.current_size = 0

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # process the last bucket
        if self.current_bucket:
            self._dispatch_bucket()

        for handle, flat_grad, grads in self.handles:
            handle.wait()
            updated_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
            for old, new in zip(grads, updated_grads):
                old.copy_(new)

        # reset
        self.handles.clear()
        self.current_bucket = []
        self.current_size = 0

    def __getattr__(self, name):
        return getattr(self.module, name)
