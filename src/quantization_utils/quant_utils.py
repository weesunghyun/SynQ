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

quant_utils.py
    - codes for utility functions for quantization

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
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

    return (pred-tgt).abs().pow(p).mean()


def find_mse_smallest(x, k, x_min=None, x_max=None):
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

def clamp(_input, _min, _max, inplace=False):
    """
    Clamp tensor input to (min, max).
    Args:
        _input: input tensor to be clamped
        _min: lower bound for clamping
        _max: upper bound for clamping
        inplace: whether to modify the input tensor in place
    """

    if inplace:
        _input.clamp_(_min, _max)
        return _input
    return torch.clamp(_input, _min, _max)


def linear_quantize(_input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    Args:
        _input: single-precision input tensor to be quantized
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(_input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(_input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        _input.mul_(scale).sub_(zero_point).round_()
        return _input
    return torch.round(scale * _input - zero_point)


def linear_dequantize(_input, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
    Args:
        _input: integer input tensor to be mapped
        scale: scaling factor for quantization
        zero_pint: shift for quantization
        inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(_input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(_input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        _input.add_(zero_point).div_(scale)
        return _input
    return (_input + zero_point) / scale


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
            ctx: context for back-propagation
            grad_output: gradient of the output
        """
        return grad_output, None, None, None

    # @staticmethod
    # def jvp(ctx, *grad_inputs): # do not use
    #     """
    #     Jacobian-vector product for quantization function.
    #     Args:
    #         ctx: context for back-propagation
    #         grad_inputs: gradient of the input
    #     """
    #     raise NotImplementedError("jvp is not implemented for quantization function.")

    # @staticmethod
    # def vjp(ctx, *grad_outputs): # do not use
    #     """
    #     Vector-Jacobian product for quantization function.
    #     Args:
    #         ctx: context for back-propagation
    #         grad_outputs: gradient of the output
    #     """
    #     raise NotImplementedError("vjp is not implemented for quantization function.")



def linear_quantize_dsg(_input, scale, zero_point, inplace=False):
    """
        Quantize single-precision input to integers with the given scaling factor and zeropoint.
        Args:
            _input: single-precision input tensor to be quantized
            scale: scaling factor for quantization
            zero_pint: shift for quantization
            inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(_input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(_input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        _input.mul_(scale).round_()
        return _input
    return torch.round(scale * _input)


def linear_dequantize_dsg(_input, scale, zero_point, inplace=False):
    """
        Map integer input tensor to fixed point float point with given scaling factor and zeropoint.
        Args:
            _input: integer input tensor to be mapped
            scale: scaling factor for quantization
            zero_pint: shift for quantization
            inplace: whether to modify the input tensor in place
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(_input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(_input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        _input.div_(scale)
        return _input
    return (_input) / scale


def symmetric_linear_quantization_params_dsg(num_bits,
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


class SymmetricQuantFunctionDSG(Function):
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

        scale, zero_point = symmetric_linear_quantization_params_dsg(
            k, x_min, x_max)
        new_quant_x = linear_quantize_dsg(x, scale, zero_point, inplace=False)
        n = 2 ** (k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize_dsg(new_quant_x,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        """
            Backward pass for quantization function.
            Args:
                ctx: context for back-propagation
                grad_output: gradient of the output
        """
        return grad_output, None, None, None

    # @staticmethod
    # def jvp(ctx, *grad_inputs): # do not use
    #     """
    #         Jacobian-vector product for quantization function.
    #         Args:
    #             ctx: context for back-propagation
    #             grad_inputs: gradient of the input
    #     """
    #     raise NotImplementedError("jvp is not implemented for quantization function.")

    # @staticmethod
    # def vjp(ctx, *grad_outputs): # do not use
    #     """
    #     Vector-Jacobian product for quantization function.
    #     Args:
    #         ctx: context for back-propagation
    #         grad_outputs: gradient of the output
    #     """
    #     raise NotImplementedError("vjp is not implemented for quantization function.")
