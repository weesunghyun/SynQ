import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization_utils.quant_modules import *

class GradCAM(object):
    def __init__(self, model_dict, verbose=False):
        self.model_arch = model_dict['arch']
        self.layer_name = model_dict.get('layer_name', None)
        self.verbose = verbose

        self.gradients = dict()
        self.activations = dict()

        self.set_target_layer()

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("Please specify size of input image in model_dict. e.g., {'input_size':(224, 224)}")
            else:
                self.generate_saliency_map_size(input_size)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]
        if torch.isnan(grad_output[0]).any():
            print("NaN detected in backward gradients")
        return None

    def forward_hook(self, module, input, output):
        self.activations['value'] = output
        if torch.isnan(output).any():
            print("NaN detected in forward activations")
        return None

    def set_target_layer(self):
        target_layer = None
        for name, module in self.model_arch.named_modules():
            if isinstance(module, (nn.Conv2d, Quant_Conv2d)):
                target_layer = module
        if target_layer is None:
            raise ValueError(type(self.model_arch), self.model_arch)
        
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def generate_saliency_map_size(self, input_size):
        device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        # print('Saliency_map size:', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        eps = 1e-10
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit.gather(1, logit.max(1)[1].view(-1, 1)).squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=retain_graph)
        gradients = self.gradients['value']
        if torch.isnan(gradients).any():
            gradients = torch.nan_to_num(gradients)
        while torch.any(gradients == 0):
            gradients += eps
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        
        assert not torch.isnan(input).any(), "Input contains NaN"
        if torch.isnan(logit).any():
            raise ValueError("NaN detected in logits before backward")
        if torch.isnan(score).any():
            raise ValueError("NaN detected in score computation")
        if torch.isnan(gradients).any():
            raise ValueError("NaN detected in gradients")
        if torch.isnan(activations).any():
            raise ValueError("NaN detected in activations")

        alpha = gradients.view(b, k, -1).mean(2).clamp(min=1e-10)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.view(b, -1).min(1)[0], saliency_map.view(b, -1).max(1)[0]
        
        saliency_map = (saliency_map - saliency_map_min.view(b, 1, 1, 1)) / (saliency_map_max.view(b, 1, 1, 1) - saliency_map_min.view(b, 1, 1, 1) + eps)

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit.gather(1, logit.max(1)[1].view(-1, 1)).squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).view(b, k, -1).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        score_expanded = score.exp().view(b, 1, 1, 1).expand_as(gradients)
        positive_gradients = F.relu(score_expanded * gradients)
        weights = (alpha * positive_gradients).view(b, k, -1).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.view(b, -1).min(1)[0], saliency_map.view(b, -1).max(1)[0]
        saliency_map = (saliency_map - saliency_map_min.view(b, 1, 1, 1)) / (saliency_map_max.view(b, 1, 1, 1) - saliency_map_min.view(b, 1, 1, 1) + 1e-7)

        return saliency_map, logit