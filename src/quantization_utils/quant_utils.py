import torch
from torch.autograd import Function

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
        loss function measured in L_p Norm
        Args:
            pred: predicted value
            tgt: target value
            p: p value for L_p Norm
            reduction: reduction method for loss
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


def find_MSESmallest(x, k, x_min=None, x_max=None):
    """
    Find the smallest MSE
    Args:
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max: upper bound for quantization range
    """

    scale, zero_point = asymmetric_linear_quantization_params(
        k, x_min, x_max)
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    n = 2 ** (k - 1)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)
    return quant_x

def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    Args:
        input: input tensor to be clamped
        min: lower bound for clamping
        max: upper bound for clamping
        inplace: whether to modify the input tensor in place
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Args:
        input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    Args:
        input: integer input tensor to be mapped
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    Args:
        num_bits: bit-setting for quantization
        saturation_min: lower bound for quantization range
        saturation_max: upper bound for quantization range
        intergral_zero_point: whether to round the zero point to the nearest integer
        signed: whether to use signed or unsigned quantization
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but not support back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        Forward pass for quantization function.
        Args:
            ctx: context for back-propagation # TODO
            x: single-precision value to be quantized
            k: bit-setting for x
            x_min: lower bound for quantization range
            x_max: upper bound for quantization range
        """

        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for quantization function.
        Args:
            ctx: context for back-propagation # TODO
            grad_output: gradient of the output
        """
        return grad_output, None, None, None

def linear_quantize_DSG(input, scale, zero_point, inplace=False):
    """
        Quantize single-precision input to integers with the given scaling factor and zeropoint.
        Args:
            input: single-precision input tensor to be quantized
            scale: scaling factor for quantization
            zero_pint: shift for quantization
            inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).round_()
        return input
    return torch.round(scale * input)


def linear_dequantize_DSG(input, scale, zero_point, inplace=False):
    """
        Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
        Args:
            input: integer input tensor to be mapped
            scale: scaling factor for quantization
            zero_pint: shift for quantization
            inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.div_(scale)
        return input
    return (input) / scale


def symmetric_linear_quantization_params_DSG(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         integral_zero_point=True,
                                         signed=True):
    """
        Compute the scaling factor and zeropoint with the given quantization range.
        Args:
            num_bits: bit-setting for quantization
            saturation_min: lower bound for quantization range
            saturation_max: upper bound for quantization range
            intergral_zero_point: whether to round the zero point to the nearest integer
            signed: whether to use signed or unsigned quantization
    """
    n = 2 ** num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


class SymmetricQuantFunction_DSG(Function):
    """
        Class to quantize the given floating-point values with given range and bit-setting.
        Currently only support inference, but not support back-propagation.
    """

    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
            Forward pass for quantization function.
            Args:
                ctx: context for back-propagation # TODO
                x: single-precision value to be quantized
                k: bit-setting for x
                x_min: lower bound for quantization range
                x_max: upper bound for quantization range
        """

        scale, zero_point = symmetric_linear_quantization_params_DSG(
            k, x_min, x_max)
        new_quant_x = linear_quantize_DSG(x, scale, zero_point, inplace=False)
        n = 2 ** (k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_DSG(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        """
            Backward pass for quantization function.
            Args:
                ctx: context for back-propagation # TODO
                grad_output: gradient of the output
        """
        return grad_output, None, None, None
