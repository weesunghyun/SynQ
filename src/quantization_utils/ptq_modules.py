import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

class AdaRoundConv2d(nn.Module):
    """Conv2d module with learnable rounding parameters for PTQ."""
    def __init__(self, weight_bit, full_precision_flag=False):
        super().__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.round_mode = 'learned'

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        self.bias = Parameter(conv.bias.data.clone()) if conv.bias is not None else None
        w = self.weight.data
        w_view = w.contiguous().view(self.out_channels, -1)
        max_abs = w_view.abs().max(dim=1)[0].view(-1, 1, 1, 1)
        n = 2 ** (self.weight_bit - 1) - 1
        scale = max_abs / n
        scale[scale == 0] = 1.0
        self.scale = Parameter(scale)
        self.alpha = Parameter(torch.zeros_like(self.weight))

    def _get_qweight(self):
        scale = self.scale
        w = self.weight / scale
        if self.training and self.round_mode == 'learned':
            w = torch.floor(w) + torch.sigmoid(self.alpha)
        else:
            w = torch.round(w)
        n = 2 ** (self.weight_bit - 1)
        w = torch.clamp(w, -n, n - 1)
        w_q = w * scale
        return w_q

    def forward(self, x):
        if self.full_precision_flag:
            w = self.weight
        else:
            w = self._get_qweight()
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class AdaRoundLinear(nn.Module):
    """Linear layer with learnable rounding parameters for PTQ."""
    def __init__(self, weight_bit, full_precision_flag=False):
        super().__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.round_mode = 'learned'

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        self.bias = Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        w = self.weight.data
        w_view = w.contiguous().view(self.out_features, -1)
        max_abs = w_view.abs().max(dim=1)[0].view(-1, 1)
        n = 2 ** (self.weight_bit - 1) - 1
        scale = max_abs / n
        scale[scale == 0] = 1.0
        self.scale = Parameter(scale)
        self.alpha = Parameter(torch.zeros_like(self.weight))

    def _get_qweight(self):
        scale = self.scale
        w = self.weight / scale
        if self.training and self.round_mode == 'learned':
            w = torch.floor(w) + torch.sigmoid(self.alpha)
        else:
            w = torch.round(w)
        n = 2 ** (self.weight_bit - 1)
        w = torch.clamp(w, -n, n - 1)
        w_q = w * scale
        return w_q

    def forward(self, x):
        if self.full_precision_flag:
            w = self.weight
        else:
            w = self._get_qweight()
        return F.linear(x, w, self.bias)
