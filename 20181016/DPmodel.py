import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _Gate(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * self.count / 2 # normalize
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)


class _Gate2(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate2, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * self.count / 2
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)

class _Gate3(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate3, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * 3 # normalize
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)

class _Gate4(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate4, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * 4 # normalize
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)

class _Gate5(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate5, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * 8 # normalize
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)


class _Bottleneck(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, count=1, gate_cls=_Gate, gate_r=4):
        super(_Bottleneck, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate,
                        kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.count = count
        self.gate = gate_cls(channels=num_input_features, reduction=gate_r, count=count)

    def forward(self, x):
        if self.count is 1:
            out = x
            x = [x]
        else:
            out = self.gate(x)

        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return x + [out]

class _Basic(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, count=1, gate_cls=_Gate, gate_r=4):
        super(_Basic, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.count = count
        self.gate = gate_cls(channels=num_input_features, reduction=gate_r, count=count)

    def forward(self, x):
        if self.count is 1:
            out = x
            x = [x]
        else:
            out = self.gate(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        return x + [out]



class _Transition(nn.Sequential):
    def __init__(self, num_input_features, count=1, gate_cls=_Gate, gate_r=4):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_input_features // 2,
                        kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gate = gate_cls(channels=num_input_features, reduction=gate_r, count=count)

    def forward(self, x):
        out = self.gate(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):

    def __init__(self, growth_rate=12,
                 num_init_features=24, num_classes=10, is_bottleneck=True, num_gate=0, layer=22):
        super(DenseNet, self).__init__()

        if layer is 22:
            block_config=[3,3,3]
        elif layer is 28:
            block_config=[4,4,4]
        elif layer is 34:
            block_config=[5,5,5]
        elif layer is 40:
            block_config=[6,6,6]

        if is_bottleneck is not True:
            block_config = [2*x for x in block_config]

        if num_gate is 0:
            gate_class = _Gate2
            gate_reduction = 24
        elif num_gate is 2:
            gate_class = _Gate2
            gate_reduction = 24
        elif num_gate is 3:
            gate_class = _Gate3
            gate_reduction = 24
        elif num_gate is 4:
            gate_class = _Gate4
            gate_reduction = 24
        elif num_gate is 5:
            gate_class = _Gate5
            gate_reduction = 24
        

        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        num_features = num_init_features

        for i in range(len(block_config)):
            layers = nn.Sequential()
            for j in range(block_config[i]):
                if is_bottleneck is True:
                    layer = _Bottleneck(num_features + j * growth_rate, growth_rate, count=j+1, gate_cls=gate_class, gate_r=gate_reduction)
                    layers.add_module('layer%d_%d' % (i + 1, j + 1), layer)
                else:
                    layer = _Basic(num_features + j * growth_rate, growth_rate, count=j+1, gate_cls=gate_class, gate_r=gate_reduction)
                    layers.add_module('layer%d_%d' % (i + 1, j + 1), layer)

            self.features.add_module('layer%d' % (i + 1), layers)
            num_features = num_features + block_config[i] * growth_rate

            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1)
                    , _Transition(num_features, count=block_config[i]+1, gate_cls=gate_class, gate_r=gate_reduction))
                num_features = num_features // 2

        # Final batch norm
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

        self.gate = gate_class(channels=num_features, reduction=gate_reduction, count=block_config[-1]+1)

        # Linear layer
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.gate(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out