"""
SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning (ICLR 2025)

Authors:
- Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
- Jongjin Kim (j2kim99@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.1

Date : Feb. 7th, 2025

Main Contact: Minjun Kim

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

conditional_batchnorm.py
    - codes for conditional batch normalization

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        """
        Initialize Conditional Batch Normalization
        Args:
            num_features: Number of features in input
            eps: Small constant to prevent division by zero
            momentum: Momentum factor applied to running mean and variance
            affine: If True, apply learned scale and shift transformation
            track_running_stats: If True, track the running mean and variance"""

        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, _input, weight, bias, **kwargs):
        """
        Forward pass of Conditional Batch Normalization
        Args:
            input: Input tensor
            weight: Weight tensor
            bias: Bias tensor
            kwargs: Additional arguments"""
        self._check_input_dim(_input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum


        output = F.batch_norm(_input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):
    """Categorical Conditional Batch Normalization"""

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        """Initialize Categorical Conditional Batch Normalization
        Args:
            num_classes: Number of classes
            num_features: Number of features in input
            eps: Small constant to prevent division by zero
            momentum: Momentum factor applied to running mean and variance
            affine: If True, apply learned scale and shift transformation
            track_running_stats: If True, track the running mean and variance"""

        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        """Initialize weights and biases"""
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, _input, c, **kwargs):
        """Forward pass of Categorical Conditional Batch Normalization"""
        weight = self.weights(c)
        bias = self.biases(c)

        return super().forward(_input, weight, bias)


class CategoricalConditionalBatchNorm2dHard(ConditionalBatchNorm2d):
    """Categorical Conditional Batch Normalization with Hard Mixing Coefficients"""
    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        """
        Initialize Categorical Conditional Batch Normalization with Hard Mixing Coefficients
        Args:
            num_classes: Number of classes
            num_features: Number of features in input
            eps: Small constant to prevent division by zero
            momentum: Momentum factor applied to running mean and variance
            affine: If True, apply learned scale and shift transformation
            track_running_stats: If True, track the running mean and variance
        """
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        """Initialize weights and biases"""
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, _input, conditions, use_mix, **kwargs):
        """
        Forward pass of Categorical Conditional Batch Normalization with Hard Mixing Coefficients
        """
        if not use_mix:
            weight = self.weights(conditions)
            bias = self.biases(conditions)
        else:
            tmp_weight = []
            tmp_bias = []
            mix_num = len(conditions[0])

            for _, ci in enumerate(conditions):
                t = self.wegights(ci[0])
                for j in range(1, len(self.weights(ci))):
                    t += self.weights(ci[j])
                tmp_weight.append(1/mix_num * t)

                t = self.biases(ci[0])
                for j in range(1, len(self.biases(ci))):
                    t += self.biases(ci[j])
                tmp_bias.append(1/mix_num*t)

            weight = torch.stack(tmp_weight, dim=0)
            bias = torch.stack(tmp_bias, dim=0)

        return super().forward(_input, weight, bias)

if __name__ == '__main__':
    pass
