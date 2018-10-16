import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _Bottleneck(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_Bottleneck, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate,
                        kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return torch.cat([x, out], 1)

class _Basic(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_Basic, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        return torch.cat([x, out], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, is_bottleneck):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            if is_bottleneck is True:
                layer = _Bottleneck(num_input_features + i * growth_rate, growth_rate)
                self.add_module('denselayer%d' % (i + 1), layer)
            else:
                layer = _Basic(num_input_features + i * growth_rate, growth_rate)
                self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_input_features // 2,
                        kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(12,12,12),
                 num_init_features=24, num_classes=10, is_bottleneck=False):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                growth_rate=growth_rate, is_bottleneck=is_bottleneck)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_input_features=num_features))
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm', nn.BatchNorm2d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module('pool', nn.AvgPool2d(kernel_size=8, stride=1))
        self.fc = nn.Linear(num_features, num_classes)

        # Linear layer
        # Official init from torch repo.

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out