import torch
import torch.nn as nn
import torchvision.models as models # Pre-Trained models

import timm     # Another Pre-trained models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x

## --------------_##
##  RESNEXT MODEL ##
## -------------- ##
from torchvision.models.resnet import ResNet, Bottleneck

# def _resnext(path, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     model.load_state_dict(torch.load(path))
#     return model
#
# def resnext101_32x16d_wsl(path, progress=True, **kwargs):
#     """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
#     and finetuned on ImageNet from Figure 5 in
#     `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
#     Args:
#         progress (bool): If True, displays a progress bar of the download to stderr.
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 16
#     return _resnext(path, Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


class MainModel:
    def __init__(self, model_type, num_classes=1):
        if model_type == 'Simple':
            self.model = SimpleModel(num_classes)
        elif model_type == 'ResNet101':
            model = models.resnet101(pretrained=False)
            model.load_state_dict(torch.load("./input/pretrained-models/resnet101-5d3b4d8f.pth"))
            for param in model.parameters():
                param.requires_grad = False
            model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024, bias=True),
                nn.Linear(in_features=1024, out_features=1, bias=True)
            )
            self.model = model
        elif model_type == 'ResNext101_32x16d':
            self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

    def get_model(self):
        return self.model

