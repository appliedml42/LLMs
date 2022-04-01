import numpy as np
import torch.optim as optim
from pytorch_lightning.utilities.cli import LR_SCHEDULER_REGISTRY


@LR_SCHEDULER_REGISTRY
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup: int,
                 total_steps: int):
        self.warmup = warmup
        self.total_steps = total_steps
        super(CosineWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        assert self.total_steps is not None
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.total_steps))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
