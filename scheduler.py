# Copyright (c) NXAI GmbH and its affiliates 2024
# Author: Maximilian Beck
# Licensed under the Apache License, Version 2.0
# Adapted from xlstm/experiments/lr_scheduler.py
# Source: https://github.com/NX-AI/xlstm

import math
from abc import abstractmethod
from torch.optim import lr_scheduler


class BaseLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self):
        """Returns the current learning rate for each parameter group."""
        raise NotImplementedError

    @abstractmethod
    def reinitialize(self, **kwargs) -> None:
        """Reinitializes the learning rate scheduler."""
        raise NotImplementedError
    
    
class LinearWarmupCosineAnnealingLR(BaseLRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_until_step, max_lr, min_lr, init_lr=0.0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_until_step = decay_until_step
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.init_lr = init_lr
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def compute_lr(step, warmup_steps, decay_until_step, max_lr, min_lr, init_lr):
        # 保证 step 非负，避免 last_epoch=-1 导致的负 lr
        step = max(0, step)

        # warmup: 从 init_lr 线性增加到 max_lr
        if step < warmup_steps:
            if warmup_steps == 0:
                return max_lr
            return init_lr + (max_lr - init_lr) * (step / warmup_steps)

        # 超过衰减截止直接返回 min_lr
        if step > decay_until_step:
            return min_lr

        # warmup 后到 decay_until_step 期间按余弦衰减（保持原逻辑）
        if warmup_steps <= step < decay_until_step:
            decay_ratio = (step - warmup_steps) / (decay_until_step - warmup_steps)
            decay_ratio = min(max(decay_ratio, 0.0), 1.0)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
        else:
            return min_lr

    def get_lr(self):
        """Returns the current learning rate for each parameter group."""
        step = self.last_epoch
        lr = self.compute_lr(step, self.warmup_steps, self.decay_until_step, self.max_lr, self.min_lr, self.init_lr)
        return [lr for _ in self.optimizer.param_groups]
