import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from base import BaseModel


class Resnet34(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()

        num_channels = 512

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.res_block1 = _build_residual_block(64, 64)

        self.res_block2 = _build_residual_block(64, 64)

        self.res_block3 = _build_residual_block(64, 64)

        self.res_block4 = _build_residual_block(64, 128, stride=2)

        self.res_block5 = _build_residual_block(128, 128)

        self.res_block6 = _build_residual_block(128, 128)

        self.res_block7 = _build_residual_block(128, 128)

        self.res_block8 = _build_residual_block(128, 256, stride=2)

        self.res_block9 = _build_residual_block(256, 256)

        self.res_block10 = _build_residual_block(256, 256)

        self.res_block11 = _build_residual_block(256, 256)

        self.res_block12 = _build_residual_block(256, 256)

        self.res_block13 = _build_residual_block(256, 256)

        self.res_block14 = _build_residual_block(256, 512, stride=2)

        self.res_block15 = _build_residual_block(512, 512)

        self.res_block16 = _build_residual_block(512, 512)

        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)

        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        assert x.shape == (x.size(0), 64, 112, 112)

        x = self.pool1(x)
        assert x.shape == (x.size(0), 64, 56, 56)

        x = _shortcut_connection(self.res_block1(x), x)
        x = _shortcut_connection(self.res_block2(x), x)
        x = _shortcut_connection(self.res_block3(x), x)
        assert x.shape == (x.size(0), 64, 56, 56)

        x = _shortcut_connection(self.res_block4(x), _zero_pad(x, 32), down_sample=True)
        x = _shortcut_connection(self.res_block5(x), x)
        x = _shortcut_connection(self.res_block6(x), x)
        x = _shortcut_connection(self.res_block7(x), x)
        assert x.shape == (x.size(0), 128, 28, 28)

        x = _shortcut_connection(self.res_block8(x), _zero_pad(x, 64), down_sample=True)
        x = _shortcut_connection(self.res_block9(x), x)
        x = _shortcut_connection(self.res_block10(x), x)
        x = _shortcut_connection(self.res_block11(x), x)
        x = _shortcut_connection(self.res_block12(x), x)
        x = _shortcut_connection(self.res_block13(x), x)
        assert x.shape == (x.size(0), 256, 14, 14)

        x = _shortcut_connection(
            self.res_block14(x), _zero_pad(x, 128), down_sample=True
        )
        x = _shortcut_connection(self.res_block15(x), x)
        x = _shortcut_connection(self.res_block16(x), x)
        assert x.shape == (x.size(0), 512, 7, 7)

        x = self.global_avg_pooling(x)
        x = squeeze(x)
        assert x.shape == (x.size(0), 512)

        return self.fc(x)


def _build_residual_block(in_channels, out_channels, stride=1, padding=1):
    return nn.Sequential(
        _build_conv_layer(in_channels, out_channels, stride, padding),
        nn.ReLU(inplace=True),
        _build_conv_layer(out_channels, out_channels),
    )


def _build_conv_layer(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
    )


def _shortcut_connection(fx, I, down_sample=False):
    if down_sample:
        return F.relu(fx + _down_sample(I))
    return F.relu(fx + I)


def _zero_pad(x, amount):
    return F.pad(x, (0, 0, 0, 0, amount, amount), "constant", 0)


def _down_sample(x):
    return F.max_pool2d(input=x, stride=2, kernel_size=2)


class TorchResnet34(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        net = models.resnet34(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)
        self.model = net

    def forward(self, x):
        return self.model(x)
