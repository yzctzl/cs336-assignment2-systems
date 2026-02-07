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
            dist.broadcast(p.data, src=0)

        # Static Bucketing
        self._build_buckets()

        # Runtime state
        self.bucket_counts = [0] * len(self.buckets)
        self.handles = []

        self._register_hooks()

    def _build_buckets(self):
        # Pre-assign parameters to buckets in REVERSE order (matching backprop)
        self.buckets: list[list[nn.Parameter]] = []
        self.param_to_bucket_idx: dict[nn.Parameter, int] = {}

        current_bucket = []
        current_size = 0

        # Iterate in reverse order of parameters
        # Note: We collect parameters that require grad
        grads_params = [p for p in self.module.parameters() if p.requires_grad]

        for p in reversed(grads_params):
            size = p.numel() * p.element_size()

            # If adding this param exceeds bucket size AND current bucket is not empty,
            # seal the current bucket.
            if current_size + size > self.bucket_size_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(p)
            current_size += size

        # Add final bucket if existing
        if current_bucket:
            self.buckets.append(current_bucket)

        # Build lookup map
        for i, bucket in enumerate(self.buckets):
            for p in bucket:
                self.param_to_bucket_idx[p] = i

    def _register_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param: nn.Parameter):
        def hook(*unused):
            idx = self.param_to_bucket_idx[param]
            self.bucket_counts[idx] += 1

            # If all params in this bucket are ready, dispatch!
            if self.bucket_counts[idx] == len(self.buckets[idx]):
                self._dispatch_bucket(idx)

        return hook

    def _dispatch_bucket(self, idx: int):
        bucket_params = self.buckets[idx]

        # flatten tensors in current bucket
        grads = [p.grad for p in bucket_params]
        flat_grad = torch._utils._flatten_dense_tensors(grads)

        # All Reduce
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG, async_op=True)

        # save for finish_gradient_synchronization
        self.handles.append((handle, flat_grad, grads))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # In static bucketing, all buckets should have been dispatched by hooks
        # (assuming all params participated in backward).
        # We just wait for handles.

        for handle, flat_grad, grads in self.handles:
            handle.wait()
            updated_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
            for old, new in zip(grads, updated_grads):
                old.copy_(new)

        # Reset state for next iteration
        self.handles.clear()
        self.bucket_counts = [0] * len(self.buckets)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
