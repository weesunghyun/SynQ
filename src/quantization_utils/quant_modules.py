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

quant_modules.py
    - codes for quantization modules

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import torch
from torch.nn import functional as F
from torch.nn import Module, Parameter

from quantization_utils.quant_utils import find_mse_smallest, lp_loss, \
    AsymmetricQuantFunction, SymmetricQuantFunctionDSG


class QuantAct(Module):
    """
        Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 beta=0.9):
        """
        Initialize the quantized activation layer
        Args:
            activation_bit: bit-setting for activation
            full_precision_flag: full precision or not
            running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return (
             f"{self.__class__.__name__}("
             f"activation_bit={self.activation_bit}, "
             f"full_precision_flag={self.full_precision_flag} "
             f"running_stat={self.running_stat}, "
             f"Act_min: {self.x_min.item()}, "
             f"Act_max: {self.x_max.item()})"
        )

    def fix(self):
        """
            fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
            fix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x):
        """
            quantize given activation x
        """

        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()

            self.beta_t = self.beta_t * self.beta
            self.x_min = (self.x_min * self.beta + x_min * (1 - self.beta))/(1 - self.beta_t)
            self.x_max = (self.x_max * self.beta + x_max * (1 - self.beta)) / (1 - self.beta_t)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max)
            return quant_act

        return x


class QuantActMSE(Module):
    """
        Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 beta=0.9):
        """
            Initialize the quantized activation layer
            Args:
                activation_bit: bit-setting for activation
                full_precision_flag: full precision or not
                running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))

        self.register_buffer('cur_x_min', torch.zeros(1))
        self.register_buffer('cur_x_max', torch.zeros(1))

        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        """
            Return the string representation of the class
        """
        return (
            f"{self.__class__.__name__}("
            f"activation_bit={self.activation_bit}, "
            f"full_precision_flag={self.full_precision_flag}, "
            f"running_stat={self.running_stat}, "
            f"Act_min: {self.x_min.item()}, "
            f"Act_max: {self.x_max.item()})"
        )

    def fix(self):
        """
            fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
            fix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x):
        """
            quantize given activation x
        """
        # print(self.running_stat, self.x_min, self.x_max)
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()

            self.cur_x_min = x_min
            self.cur_x_max = x_max

            save_min = None
            save_max = None

            x_clone = x.clone().detach()
            # in-place operation used on multi-gpus
            # self.x_min += -self.x_min + min(self.x_min, x_min)
            # self.x_max += -self.x_max + max(self.x_max, x_max)
            best_score = 1e+10
            # print('mse find best max/min')
            for i in range(80):
                new_min = x_min * (1.0 - (i * 0.01))
                new_max = x_max * (1.0 - (i * 0.01))

                quant_act = find_mse_smallest(x_clone, self.activation_bit, new_min, new_max)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x_clone, quant_act, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    save_min = new_min
                    save_max = new_max

            self.beta_t = self.beta_t * self.beta
            self.x_min = self.x_min * self.beta + save_min * (1 - self.beta)
            self.x_max = self.x_max * self.beta + save_max * (1 - self.beta)
            # print(self.x_min, self.x_max, save_min, save_max, x.data.min(),  x.data.max())

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max)
            return quant_act

        return x


class QuantLinear(Module):
    """
        Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
          Initialize the quantized linear layer
          Args:
            weight_bit: bit-setting for weight
            full_precision_flag: full precision or not
            running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        """
            Return the string representation of the class
        """
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, full_precision_flag={self.full_precision_flag})"
        return s

    def set_param(self, linear):
        """
            Set the parameters of the quantized linear layer
        """
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
            using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class QuantConv2d(Module):
    """
        Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
            Initialize the quantized convolutional layer
            Args:
                weight_bit: bit-setting for weight
                full_precision_flag: full precision or not
        """
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        """
            Return the string representation of the class
        """
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, full_precision_flag={self.full_precision_flag})"
        return s

    def set_param(self, conv):
        """
            Set the parameters of the convolutional layer
            Args:
                conv: the convolutional layer
        """
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QuantActDSG(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True,
                 beta=0.9):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))

        self.act_function = SymmetricQuantFunctionDSG.apply

    def __repr__(self):
        """
            Return the string representation of the class
        """
        return (
             f"{self.__class__.__name__}("
             f"activation_bit={self.activation_bit}, "
             f"full_precision_flag={self.full_precision_flag} "
             f"running_stat={self.running_stat}, "
             f"Act_min: {self.x_min.item()}, "
             f"Act_max: {self.x_max.item()}"
        )

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            # self.x_min += -self.x_min + min(self.x_min, x_min)
            # self.x_max += -self.x_max + max(self.x_max, x_max)
            if x_min.abs() > x_max.abs():
                x_min = -x_min.abs()
                x_max = x_min.abs()
            else:
                x_min = -x_max.abs()
                x_max = x_max.abs()
            self.beta_t = self.beta_t * self.beta
            self.x_min = (self.x_min * self.beta + x_min *
                          (1 - self.beta))/(1 - self.beta_t)
            self.x_max = (self.x_max * self.beta + x_max *
                          (1 - self.beta)) / (1 - self.beta_t)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min,
                                          self.x_max)
            return quant_act

        return x


class QuantLinearDSG(Module):
    """
    Class to quantize given linear layer weights
    """

    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = SymmetricQuantFunctionDSG.apply

    def __repr__(self):
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, full_precision_flag={self.full_precision_flag})"
        return s

    def set_param(self, linear):
        """
            Set the parameters of the linear layer
            Args:
                linear: the linear layer
        """
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = -x_transform.abs().max(dim=1).values
        w_max = x_transform.abs().max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class QuantConv2dDSG(Module):
    """
        Class to quantize given convolutional layer weights
    """

    def __init__(self, weight_bit, full_precision_flag=False):
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = SymmetricQuantFunctionDSG.apply

    def __repr__(self):
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, full_precision_flag={self.full_precision_flag})"
        return s

    def set_param(self, conv):
        """
        Set the parameters of the convolutional layer
        Args:
            conv: the convolutional layer
        """
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = -x_transform.abs().max(dim=1).values
        w_max = x_transform.abs().max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

if __name__ == "__main__":
    m = QuantAct(8)
    print(m)
