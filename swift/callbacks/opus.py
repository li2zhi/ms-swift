import torch

from typing import TYPE_CHECKING

from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import Trainer, TrainingArguments


class OpusCallback(TrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.ema = 0.9


    def on_optimizer_step(self, args, state, control, **kwargs):
        # use batch grad_norm
        indices = self.trainer.sample_indices
        self.trainer.sample_indices = []

        grad_norm = self._preconditioned_grad_norm()
        self.trainer.sampler.update(indices, [grad_norm / len(indices)] * len(indices), self.ema)


    def _preconditioned_grad_norm(self):
        total = 0.0

        for p in self.trainer.model.parameters():
            if p.grad is None:
                continue

            state = self.trainer.optimizer.state.get(p)
            if state is None:
                continue

            v = state.get("exp_avg_sq")
            if v is None:
                continue

            g = p.grad
            denom = torch.sqrt(v) + 1e-8
            precond = g / denom
            total += (precond ** 2).sum().item()

        return total ** 0.5
