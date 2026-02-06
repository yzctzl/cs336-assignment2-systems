import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles: list[dist.Work] = []

        # sync_ddp_parameters
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        self._register_hooks()

    def _register_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:

                def hook_fn(param):
                    handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
                    self.handles.append(handle)  # pyright: ignore[reportArgumentType]

                p.register_post_accumulate_grad_hook(hook_fn)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
