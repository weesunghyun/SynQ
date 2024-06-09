"""
    # TODO: add description
"""

import torch
from torch import nn
from torch.nn import functional as F

from quantization_utils.quant_modules import QuantConv2d

class GradCAM:
    """Calculate GradCAM salinecy map."""
    def __init__(self, model_dict, verbose=False):
        """
        GradCAM constructor
        Args:
            model_dict: Dictionary containing model architecture, input_size, and layer_name
            verbose: Print saliency map size
        """

        self.model_arch = model_dict['arch']
        self.layer_name = model_dict.get('layer_name', None)
        self.verbose = verbose

        self.gradients = {}
        self.activations = {}

        self.set_target_layer()

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("Please specify size of input image in model_dict. \
                      e.g., {'input_size':(224, 224)}")
            else:
                self.generate_saliency_map_size(input_size)

    def backward_hook(self, module, grad_input, grad_output):
        """
        Hook to store gradients of the target layer
        Args:
            module: Target layer
            grad_input: Gradients of the input
            grad_output: Gradients of the output
        """

        self.gradients['value'] = grad_output[0]

        if torch.isnan(grad_output[0]).any():
            for idx, grad in enumerate(grad_input):
                if torch.isnan(grad).any():
                    print(f"grad_input[{idx}] contains NaN")

    def forward_hook(self, module, _input, output):
        """
        Hook to store activations of the target layer
        Args:
            module: Target layer
            _input: Input of the target layer
            output: Output of the target layer
        """

        self.activations['value'] = output
        if torch.isnan(output).any():
            print("NaN detected in forward activations")

    def set_target_layer(self):
        """
        Set target layer to the last convolutional layer
        """

        target_layer = None
        for _, module in self.model_arch.named_modules():
            if isinstance(module, (nn.Conv2d, QuantConv2d)):
                target_layer = module

        if target_layer is None:
            raise ValueError(type(self.model_arch), self.model_arch)

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def generate_saliency_map_size(self, input_size):
        """
        Generate saliency map size
        Args:
            input_size: Size of the input image
        """
        device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))

    def forward(self, _input, class_idx=None, retain_graph=False):
        """
        Forward pass of the input image
        Args:
            _input: Input image
            class_idx: Index of the class
        """
        b, _, h, w = _input.size()
        eps = 1e-10

        if torch.isnan(_input).any():
            print(f"NaN detected in input before logits: {_input.shape}")
            raise ValueError("NaN detected in input before backward")

        for name, param in self.model_arch.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                print(f"Parameter stats: min={param.min().item()}, \
                      max={param.max().item()}, \
                      mean={param.mean().item()}")
                raise ValueError(f"NaN detected in parameter: {name}")

            if torch.isinf(param).any():
                print(f"Inf detected in parameter: {name}")
                print(f"Parameter stats: min={param.min().item()}, \
                      max={param.max().item()}, \
                      mean={param.mean().item()}")
                raise ValueError(f"Inf detected in parameter: {name}")

        logit = self.model_arch(_input)

        if torch.isnan(logit).any():
            raise ValueError("NaN detected in logits before backward")

        if class_idx is None:
            score = logit.gather(1, logit.max(1)[1].view(-1, 1)).squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        if torch.isnan(score).any():
            print(f"NaN detected in score: {score}")
            raise ValueError("NaN detected in score computation")

        self.model_arch.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=retain_graph)
        gradients = self.gradients['value']

        if torch.isnan(gradients).any():
            print("NaN detected in gradients before processing")
            gradients = torch.nan_to_num(gradients)
            if torch.isnan(gradients).any():
                raise ValueError("NaN detected in gradients after processing")

        while torch.any(gradients == 0):
            gradients += eps
        activations = self.activations['value']

        if torch.isnan(activations).any():
            print("NaN detected in activations")
            raise ValueError("NaN detected in activations")

        b, k, _, _ = gradients.size()

        assert not torch.isnan(input).any(), "Input contains NaN"

        alpha = gradients.view(b, k, -1).mean(2).clamp(min=1e-10)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map,
                                     size=(h, w),
                                     mode='bilinear',
                                     align_corners=False)

        saliency_map_min = saliency_map.view(b, -1).min(1)[0]
        saliency_map_max = saliency_map.view(b, -1).max(1)[0]

        saliency_map -= saliency_map_min.view(b, 1, 1, 1)
        saliency_map /= saliency_map_max.view(b, 1, 1, 1) - saliency_map_min.view(b, 1, 1, 1) + eps

        return saliency_map, logit

    def __call__(self, _input, class_idx=None, retain_graph=False):
        """
        Call method
        Args:
            _input: Input image
            class_idx: Index of the class
        """

        return self.forward(_input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """
    Calculate GradCAM++ saliency map.
    """
    # def __init__(self, model_dict, verbose=False):
    #     """
    #     GradCAM++ constructor
    #     Args:
    #         model_dict: Dictionary containing model architecture, input_size, and layer_name
    #         verbose: Print saliency map size
    #     """
    #     super().__init__(model_dict, verbose)

    def forward(self, _input, class_idx=None, retain_graph=False):
        """
        Forward pass of the input image
        Args:
            _input: Input image
            class_idx: Index of the class
        """
        b, _, h, w = _input.size()

        logit = self.model_arch(_input)
        if class_idx is None:
            score = logit.gather(1, logit.max(1)[1].view(-1, 1)).squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, _, _ = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, -1).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        score_expanded = score.exp().view(b, 1, 1, 1).expand_as(gradients)
        positive_gradients = F.relu(score_expanded * gradients)
        weights = (alpha * positive_gradients).view(b, k, -1).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w),
                                     mode='bilinear',
                                     align_corners=False)

        saliency_map_min = saliency_map.view(b, -1).min(1)[0]
        saliency_map_max = saliency_map.view(b, -1).max(1)[0]

        saliency_map = (saliency_map - saliency_map_min.view(b, 1, 1, 1)) / \
            (saliency_map_max.view(b, 1, 1, 1) - saliency_map_min.view(b, 1, 1, 1) + 1e-7)

        return saliency_map, logit
