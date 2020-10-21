import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class Resnet34(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        
        ## num_flatten_params = 512 * 1 * 1
        num_channels = 512
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)
        
        
        self.res_block1 = self._build_residual_block(64, 64)
        
        self.res_block2 = self._build_residual_block(64, 64)
        
        self.res_block3 = self._build_residual_block(64, 64)
        
        self.res_block4 = self._build_residual_block(64, 128, stride=2)
        
        self.res_block5 = self._build_residual_block(128, 128)
        
        self.res_block6 = self._build_residual_block(128, 128)
        
        self.res_block7 = self._build_residual_block(128, 128)
        
        self.res_block8 = self._build_residual_block(128, 256, stride=2)
        
        self.res_block9 = self._build_residual_block(256, 256)
        
        self.res_block10 = self._build_residual_block(256, 256)
        
        self.res_block11 = self._build_residual_block(256, 256)
        
        self.res_block12 = self._build_residual_block(256, 256)
        
        self.res_block13 = self._build_residual_block(256, 256)
        
        self.res_block14 = self._build_residual_block(256, 512, stride=2)
        
        self.res_block15 = self._build_residual_block(512, 512)
        
        self.res_block16 = self._build_residual_block(512, 512)
        
        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        
        self.fc = nn.Linear(num_channels, num_classes)
        
    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        assert(x.shape == (x.size(0), 64, 112, 112))
        
        x = self.pool1(x)
        assert(x.shape == (x.size(0), 64, 56, 56))
        
        x = F.relu(self.res_block1(x) + x)
        x = F.relu(self.res_block2(x) + x)
        x = self.res_block3(x) + x
        assert(x.shape == (x.size(0), 64, 56, 56))
        
        ## print(f"shape of x before zero pad: {x.shape}")
        ## print(f"shape of x applying zero pad: {self._zero_pad(x, 32).shape}")
        
        x = F.relu(self.res_block4(x) + self._down_sample(self._zero_pad(x, 32)))
        x = F.relu(self.res_block5(x) + x)
        x = F.relu(self.res_block6(x) + x)
        x = F.relu(self.res_block7(x) + x)
        assert(x.shape == (x.size(0), 128, 28, 28))
        
        x = F.relu(self.res_block8(x) + self._down_sample(self._zero_pad(x, 64)))
        x = F.relu(self.res_block9(x) + x)
        x = F.relu(self.res_block10(x) + x)
        x = F.relu(self.res_block11(x) + x)
        x = F.relu(self.res_block12(x) + x)
        x = F.relu(self.res_block13(x) + x)
        assert(x.shape == (x.size(0), 256, 14, 14))
        
        x = F.relu(self.res_block14(x) + self._down_sample(self._zero_pad(x, 128)))
        x = F.relu(self.res_block15(x) + x)
        x = F.relu(self.res_block16(x) + x)
        assert(x.shape == (x.size(0), 512, 7, 7))
        
        ## print(f"shape of x before g_avg_pooling: {x.shape}")
        x = self.global_avg_pooling(x)
        ## print(f"shape of x after g_avg_pooling: {x.shape}")
        x = squeeze(x)
        ## print(f"shape of x after squeeze: {x.shape}")
        assert(x.shape == (x.size(0), 512))
        
        return self.fc(x)
        
    def _build_residual_block(self, in_channels, out_channels, stride=1, padding=1):
        return nn.Sequential(
            self._build_conv_layer(in_channels, out_channels, stride, padding),
            nn.ReLU(inplace=True),
            self._build_conv_layer(out_channels, out_channels),
        )
    
    def _down_sample(self, x):
        return F.max_pool2d(input=x, stride=2, kernel_size=2)
    
    def _zero_pad(self, x, amount):
        return F.pad(x, (0, 0, 0, 0, amount, amount), "constant", 0)
    
    def _build_conv_layer(self, in_channels, out_channels, stride=1, padding=1):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding)
    
    def _shortcut_connection(fx, I):
        return F.relu(fx + I)
    
    
    def _flatten(self, x):
        return x.view(x.size(0) - 1)
    