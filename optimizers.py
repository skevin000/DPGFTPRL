import torch


__all__ = ['DPGroupFTRLProximalOptimizer']

class DPGroupFTRLProximalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, alpha=1.0, beta=0.1, l1_regs=None, momentum=0.0, noise_multiplier=0.0, clip_norm=0.0):
        if l1_regs is None:
            l1_regs = [0.01] * len(params)
        assert len(l1_regs) == len(params), "Each parameter group must have an associated L1 regularization strength"

        super().__init__(params, dict(lr=lr, alpha=alpha, beta=beta, momentum=momentum, noise_multiplier=noise_multiplier, clip_norm=clip_norm))
        self.l1_regs = l1_regs

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group_idx, group in enumerate(self.param_groups):
            alpha, beta, l1_reg, momentum = group['alpha'], group['beta'], self.l1_regs[group_idx], group['momentum']
            noise_multiplier, clip_norm = group['noise_multiplier'], group['clip_norm']

            for p in group['params']:
                if p.grad is None: continue

                state = self.state[p]
                if not state:
                    state['grad_sum'] = torch.zeros_like(p.data)
                    state['model_sum'] = p.data.clone()
                    state['momentum'] = torch.zeros_like(p.data)

                grad_sum, model_sum, momentum_vec = state['grad_sum'], state['model_sum'], state['momentum']
                
                # Gradient clipping
                grad_norm = p.grad.data.norm()
                if grad_norm > clip_norm:
                    p.grad.data.mul_(clip_norm / grad_norm)

                # Add noise to the gradient
                noise = torch.randn_like(p.grad.data) * noise_multiplier * clip_norm

                grad_sum.add_(p.grad * alpha, alpha=group['lr'])
                model_sum.add_(p.data - grad_sum.sign() * l1_reg, alpha=group['lr'])

                if momentum > 0:
                    momentum_vec.mul_(momentum).add_(p.grad, alpha=group['lr'])
                    p.data.copy_(model_sum - momentum_vec + noise)

        return loss

    @torch.no_grad()
    def restart(self, last_noise=None):
        assert last_noise is not None or hasattr(self, 'record_last_noise')

        for group in self.param_groups:
            for p, nz in zip(group['params'], last_noise if last_noise else self.record_last_noise):
                if p.grad is None: continue
                self.state[p]['grad_sum'].add_(nz)
