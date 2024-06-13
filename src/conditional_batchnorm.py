import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

class ConditionalBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else: 
                exponential_average_factor = self.momentum


        output = F.batch_norm(input, self.running_mean, self.running_var,
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

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


class CategoricalConditionalBatchNorm2d_hard(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d_hard, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, use_mix, **kwargs):
        if not use_mix:
            weight = self.weights(c)
            bias = self.biases(c)
        else:
            tmp_weight = []
            tmp_bias = []
            mix_num = len(c[0])

            for i in range(len(c)):
                t = self.weights(c[i][0])
                for j in range(1, len(self.weights(c[i]))):
                    t += self.weights(c[i][j])
                tmp_weight.append(1/mix_num * t)

                t = self.biases(c[i][0])
                for j in range(1, len(self.biases(c[i]))):
                    t += self.biases(c[i][j])
                tmp_bias.append(1/mix_num * t)


            weight = torch.stack(tmp_weight, dim=0)
            bias = torch.stack(tmp_bias, dim=0)
            print(weight[0])


        return super(CategoricalConditionalBatchNorm2d_hard, self).forward(input, weight, bias)

if __name__ == '__main__':
    pass