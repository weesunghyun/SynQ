import numpy as np

import cv2

import torch


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """
    Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name: the name of layer with its hierarchical information.
            please refer to usages below.

            - 'conv1'
            - 'layer1'
            - 'layer1_basicblock0'
            - 'layer1_basicblock0_relu'
            - 'layer1_bottleneck0'
            - 'layer1_bottleneck0_conv1'
            - 'layer1_bottleneck0_downsample'
            - 'layer1_bottleneck0_downsample_0'
            - 'avgpool'
            - 'fc'

    Return:
        target_layer: found layer which will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError(f'unknown layer : {target_layer_name}')

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    elif 'stage4' in target_layer_name:
        # raise KeyError(arch.module.features.stage4)
        target_layer = arch.module.features.stage4

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """
    Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name: the name of layer with its hierarchical information.
            please refer to usages below.

            - 'features'
            - 'features_transition1'
            - 'features_transition1_norm'
            - 'features_denseblock2_denselayer12'
            - 'features_denseblock2_denselayer12_norm1'
            - 'features_denseblock2_denselayer12_norm1'
            - 'classifier'

    Return:
        target_layer: found layer which will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """
    Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name: the name of layer with its hierarchical information.
            please refer to usages below.

            - 'features'
            - 'features_42'
            - 'classifier'
            - 'classifier_0'

    Return:
        target_layer: found layer which will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """
    Find alexnet layer to calculate GradCAM and GradCAM++.

    Args:
        arch: default torchvision densenet models.
        target_layer_name: The name of the layer with its hierarchical information.
            Please refer to usages below:

            - 'features'
            - 'features_0'
            - 'classifier'
            - 'classifier_0'

    Returns:
        target_layer: The found layer which will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """
    Find squeezenet layer to calculate GradCAM and GradCAM++.

    Args:
        arch: Default torchvision densenet models.
        target_layer_name: The name of the layer with its hierarchical information.
            Please refer to usages below:

            - 'features_12'
            - 'features_12_expand3x3'
            - 'features_12_expand3x3_activation'

    Returns:
        target_layer: The found layer which will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Returns:
        Tensor: Denormalized Tensor image.
    """
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    """
    Normalize a tensor image with mean and standard deviation.

    Args:
        tensor: Tensor image of size (C, H, W) to be normalized.
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
            Initialize Normalize transform.
            Args:
                mean: Sequence of means for each channel.
                std: Sequence of standard deviations for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        """
        Normalize a tensor image with mean and standard deviation.

        Args:
            tensor: Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        """
        Denormalize a tensor image with mean and standard deviation.

        Args:
            tensor: Tensor image of size (C, H, W) to be denormalized.

        Returns:
            Tensor: Denormalized Tensor image.
        """
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
