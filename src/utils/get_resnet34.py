"""
    # TODO: add description
"""

import torch
from torch import nn
from torch.nn import init

from pytorchcv.models.resnet import ResUnit
from pytorchcv.models.common import conv3x3_block

class CIFARResNet(nn.Module):
    """
        ResNet model for CIFAR from 'Deep Residual Learning for Image Recognition'
        Reference: https://arxiv.org/abs/1512.03385.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        """
            Initialize the model.
            Args:
                channels: number of output channels for each unit.
                init_block_channels: number of output channels for the initial unit.
                bottleneck: whether to use a bottleneck or simple block in units.
                in_channels: number of input channels.
                in_size: spatial size of the expected input image.
                num_classes: number of classification classes.
        """
        super().__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module(f"unit{j+1}", ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=False))
                in_channels = out_channels
            self.features.add_module(f"stage{i+1}", stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d((1, 1)))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        """
            Initialize model parameters.
        """
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        """
            Forward pass the model.
            Args:
                x: the input data
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def resnet34_get_model():
    """
        Get the ResNet-34 model for CIFAR-100 dataset
    """

    channels = [[64, 64, 64],
                [128, 128, 128, 128],
                [256, 256, 256, 256, 256, 256],
                [512, 512, 512]]

    net = CIFARResNet(channels= channels,
                       init_block_channels=64,
                       bottleneck=None,
                       in_channels=3,
                       in_size=(32,32),
                       num_classes=100)
    net.load_state_dict(torch.load('./checkpoints/resnet34.pth'))
    net.cuda()

    return net

if __name__ == '__main__':
    model = resnet34_get_model()

    dummy = torch.randn(1, 3, 32, 32).cuda()

    out = model(dummy)
    print(f"Inference OK! {out.shape}")
