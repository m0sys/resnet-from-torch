from typing import Optional
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from base import BaseModel


class Resnet(BaseModel):
    def __init__(
        self, num_classes: Optional[int] = 100, num_features: Optional[int] = 2048
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)

        self.fc = nn.Linear(self.num_features, self.num_classes)


class Resnet34(Resnet):
    def __init__(self, num_classes: Optional[int] = 100):
        super().__init__(num_classes=num_classes, num_features=512)

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

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
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


class Resnet50(Resnet):
    def __init__(self, num_classes: Optional[int] = 100):
        super().__init__(num_classes=num_classes, num_features=2048)

        self.res_block1 = _build_residual_bottleneck_block(
            in_channels=64, bn_channels=64, out_channels=256
        )
        self.res_block2 = _build_residual_bottleneck_block(
            in_channels=256, bn_channels=64, out_channels=256
        )
        self.res_block3 = _build_residual_bottleneck_block(
            in_channels=256, bn_channels=64, out_channels=256
        )

        self.res_block4 = _build_residual_bottleneck_block(
            in_channels=256, bn_channels=128, out_channels=512, stride=2
        )
        self.res_block5 = _build_residual_bottleneck_block(
            in_channels=512, bn_channels=128, out_channels=512
        )
        self.res_block6 = _build_residual_bottleneck_block(
            in_channels=512, bn_channels=128, out_channels=512
        )
        self.res_block7 = _build_residual_bottleneck_block(
            in_channels=512, bn_channels=128, out_channels=512
        )

        self.res_block8 = _build_residual_bottleneck_block(
            in_channels=512, bn_channels=256, out_channels=1024, stride=2
        )
        self.res_block9 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=256, out_channels=1024
        )
        self.res_block10 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=256, out_channels=1024
        )
        self.res_block11 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=256, out_channels=1024
        )
        self.res_block12 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=256, out_channels=1024
        )
        self.res_block13 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=256, out_channels=1024
        )

        self.res_block14 = _build_residual_bottleneck_block(
            in_channels=1024, bn_channels=512, out_channels=2048, stride=2
        )
        self.res_block15 = _build_residual_bottleneck_block(
            in_channels=2048, bn_channels=512, out_channels=2048
        )
        self.res_block16 = _build_residual_bottleneck_block(
            in_channels=2048, bn_channels=512, out_channels=2048
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        assert x.shape == (x.size(0), 64, 112, 112)

        x = self.pool1(x)
        assert x.shape == (x.size(0), 64, 56, 56)

        x = _shortcut_connection(self.res_block1(x), _zero_pad(x, 96))
        x = _shortcut_connection(self.res_block2(x), x)
        x = _shortcut_connection(self.res_block3(x), x)
        assert x.shape == (x.size(0), 256, 56, 56)

        x = _shortcut_connection(
            self.res_block4(x), _zero_pad(x, 128), down_sample=True
        )
        x = _shortcut_connection(self.res_block5(x), x)
        x = _shortcut_connection(self.res_block6(x), x)
        x = _shortcut_connection(self.res_block7(x), x)
        assert x.shape == (x.size(0), 512, 28, 28)

        x = _shortcut_connection(
            self.res_block8(x), _zero_pad(x, 256), down_sample=True
        )
        x = _shortcut_connection(self.res_block9(x), x)
        x = _shortcut_connection(self.res_block10(x), x)
        x = _shortcut_connection(self.res_block11(x), x)
        x = _shortcut_connection(self.res_block12(x), x)
        x = _shortcut_connection(self.res_block13(x), x)
        assert x.shape == (x.size(0), 1024, 14, 14)

        x = _shortcut_connection(
            self.res_block14(x), _zero_pad(x, 512), down_sample=True
        )
        x = _shortcut_connection(self.res_block15(x), x)
        x = _shortcut_connection(self.res_block16(x), x)
        assert x.shape == (x.size(0), 2048, 7, 7)

        x = self.global_avg_pooling(x)
        x = squeeze(x)
        assert x.shape == (x.size(0), self.num_features)

        return self.fc(x)


class TorchResnet34(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        net = models.resnet34(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)
        self.model = net

    def forward(self, x):
        return self.model(x)


class TorchResnet50(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        net = models.resnet50(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)
        self.model = net

    def forward(self, x):
        return self.model(x)


class Cifar10Resnet(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        self.res_block1 = _build_residual_block(3, 16)
        self.res_block2 = _build_residual_block(16, 32)
        self.res_block3 = _build_residual_block(32, 64)

        self.global_avg_pooling = nn.AvgPool2d(kernel_size=8)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = _shortcut_connection(self.res_block1(x), _zero_pad(x, 8), down_sample=True)

        x = _shortcut_connection(self.res_block2(x), _zero_pad(x, 16), down_sample=True)

        x = _shortcut_connection(self.res_block3(x), _zero_pad(x, 32), down_sample=True)

        x = self.global_avg_pooling(x)

        x = squeeze(x)

        return self.fc(x)


def _build_residual_block(in_channels, out_channels, stride=1, padding=1):
    return nn.Sequential(
        _build_conv_layer(in_channels, out_channels, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        _build_conv_layer(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
    )


def _shortcut_connection(fx, I, down_sample=False):
    if down_sample:
        return F.relu(fx + _down_sample(I))
    return F.relu(fx + I)


def _zero_pad(x, amount):
    return F.pad(x, (0, 0, 0, 0, amount, amount), "constant", 0)


def _down_sample(x):
    return F.max_pool2d(input=x, stride=2, kernel_size=2)


def _build_conv_layer(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
    )


def _build_residual_bottleneck_block(
    in_channels: int,
    bn_channels: int,
    out_channels: int,
    stride: Optional[int] = 1,
    padding: Optional[int] = 1,
):
    return nn.Sequential(
        _build_bottleneck_conv_layer(in_channels, bn_channels),
        nn.BatchNorm2d(bn_channels),
        nn.ReLU(inplace=True),
        _build_bottleneck_conv_layer(bn_channels, bn_channels),
        nn.BatchNorm2d(bn_channels),
        nn.ReLU(inplace=True),
        _build_conv_layer(bn_channels, out_channels, stride, padding),
        nn.BatchNorm2d(out_channels),
    )


def _build_bottleneck_conv_layer(
    in_channels: int,
    out_channels: int,
):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )
