import torch
import torch.nn as nn
import torch.nn.functional as F

class _Gate(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        m_channel = channels * count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(m_channel, m_channel//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(m_channel//reduction, self.count, bias=False)
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
            self.p[i] = self.p[i] / p_sum * self.count
            x[i] = x[i] * self.p[i]

        return sum(x)


class _Gate2(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate2, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        m_channel = channels * count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(m_channel, reduction, bias=False)
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
            self.p[i] = self.p[i] / p_sum * self.count
            x[i] = x[i] * self.p[i]

        return sum(x)

class BasicBlock(nn.Module):

    def __init__(self, channels, init_block=False, count=1, num_gate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.init_block = init_block
        
        if num_gate is 0:
            self.gate = _Gate(channels=channels, reduction=4, count=count)
        elif num_gate is 1:
            self.gate = _Gate(channels=channels, reduction=2, count=count)
        elif num_gate is 2:
            self.gate = _Gate2(channels=channels, reduction=16, count=count)
        elif num_gate is 3:
            self.gate = _Gate2(channels=channels, reduction=24, count=count)
        elif num_gate is 4:
            self.gate = _Gate2(channels=channels, reduction=32, count=count)


    def forward(self, x):
        if self.init_block is True:
            out = x
            x = [x]
        else:
            # out = sum(x) # change to weighted sum
            out = self.gate(x)
            out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return x + [out]

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, count=2, num_gate=0):
        super(_Transition, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if num_gate is 0:
            self.gate = _Gate(channels=in_channels, reduction=4, count=count)
        elif num_gate is 1:
            self.gate = _Gate(channels=in_channels, reduction=2, count=count)
        elif num_gate is 2:
            self.gate = _Gate2(channels=in_channels, reduction=16, count=count)
        elif num_gate is 3:
            self.gate = _Gate2(channels=in_channels, reduction=24, count=count)
        elif num_gate is 4:
            self.gate = _Gate2(channels=in_channels, reduction=32, count=count)
        
    def forward(self, x):

        # out = sum(x)
        out = self.gate(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return [out]

class ResNet(nn.Module):
    def __init__(self, num_classes=10, resnet_layer=56, num_gate=0):
        super(ResNet, self).__init__()

        self.features = nn.Sequential()
        num_features = 16

        self.features.add_module('conv1', nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn1', nn.BatchNorm2d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        if resnet_layer is 14:
            block_config = (6,6,6)
        elif resnet_layer is 20:
            block_config = (6,6,6)
        elif resnet_layer is 32:
            block_config = (6,6,6)
        elif resnet_layer is 44:
            block_config = (6,6,6)
        elif resnet_layer is 56:
            block_config = (6,6,6)
        elif resnet_layer is 110:
            block_config = (6,6,6)
            
        self.features.add_module('first_layer', BasicBlock(channels=num_features, init_block=True))
        layer = nn.Sequential()
        add_on = 2
        for i in range(block_config[0]):
            layer.add_module('layer%d_%d' % (1, i), BasicBlock(channels=num_features, init_block=False, count=i+add_on, num_gate=num_gate))
            
        
        self.features.add_module('layer%d' % (1), layer)
        self.features.add_module('transition%d' % (1), _Transition(in_channels=num_features
                , out_channels=num_features * 2, count=block_config[0]+add_on, num_gate=num_gate))
        num_features = num_features * 2


        layer2 = nn.Sequential()
        add_on = 1
        for i in range(block_config[1]):
            layer2.add_module('layer%d_%d' % (2, i), BasicBlock(channels=num_features, init_block=False, count=i+add_on, num_gate=num_gate))
        
        self.features.add_module('layer%d' % (2), layer2)
        self.features.add_module('transition%d' % (2), _Transition(in_channels=num_features
                , out_channels=num_features * 2, count=block_config[1]+add_on, num_gate=num_gate))
        num_features = num_features * 2

       
        layer3 = nn.Sequential()
        add_on = 1
        for i in range(block_config[2]):
            layer3.add_module('layer%d_%d' % (3, i), BasicBlock(channels=num_features, init_block=False, count=i+add_on, num_gate=num_gate))
        
        self.features.add_module('layer%d' % (3), layer3)

        add_on = 1
        # Final batch norm
        if num_gate is 0:
            self.gate = _Gate(channels=num_features, reduction=4, count=block_config[-1] + add_on)
        elif num_gate is 1:
            self.gate = _Gate(channels=num_features, reduction=2, count=block_config[-1] + add_on)
        elif num_gate is 2:
            self.gate = _Gate2(channels=num_features, reduction=16, count=block_config[-1] + add_on)
        elif num_gate is 3:
            self.gate = _Gate2(channels=num_features, reduction=24, count=block_config[-1] + add_on)
        elif num_gate is 4:
            self.gate = _Gate2(channels=num_features, reduction=32, count=block_config[-1] + add_on)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = self.features(x)
        # out = sum(out)
        out = self.gate(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out