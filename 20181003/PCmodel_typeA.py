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

    def __init__(self, in_channels, out_channels, stride, downsample=None, init_block=False, count=1, num_gate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.init_block = init_block

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
        
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), #avgPooling?
                nn.BatchNorm2d(out_channels)
            )

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

        if self.downsample is not None:
            x_ = []
            for item in x:
                x_.append(self.downsample(item))
            x = x_            

        return x + [out]

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
            

        count = 1
        for i, num_layers in enumerate(block_config):
            if i is 0:
                layer = nn.Sequential()
                layer.add_module('layer%d_0' % (i+1), BasicBlock(in_channels=num_features
                        , out_channels=num_features, stride=1, downsample=None, init_block=True, num_gate=num_gate))
                count += 1
            else:
                layer = nn.Sequential()
                layer.add_module('layer%d_0' % (i+1), BasicBlock(in_channels=num_features
                        , out_channels=num_features*2, stride=2, downsample=True, init_block=False, count=count, num_gate=num_gate))
                num_features = num_features * 2
                count += 1

            for j in range(1, num_layers):
                layer.add_module('layer%d_%d' % (i+1, j), BasicBlock(in_channels=num_features
                        , out_channels=num_features, stride=1, downsample=None, init_block=False, count=count, num_gate=num_gate))
                count += 1


            self.features.add_module('layer%d' % (i + 1), layer)
            # if i != len(block_config) - 1:
            #     self.features.add_module('transition%d' % (i + 1), _Transition(in_channels=num_features, out_channels=num_features * 2))
            #     num_features = num_features * 2

        # Final batch norm
        if num_gate is 0:
            self.gate = _Gate(channels=num_features, reduction=4, count=count)
        elif num_gate is 1:
            self.gate = _Gate(channels=num_features, reduction=2, count=count)
        elif num_gate is 2:
            self.gate = _Gate2(channels=num_features, reduction=16, count=count)
        elif num_gate is 3:
            self.gate = _Gate2(channels=num_features, reduction=24, count=count)
        elif num_gate is 4:
            self.gate = _Gate2(channels=num_features, reduction=32, count=count)
        # elif num_gate is 5:
        # elif num_gate is 6:
        # elif num_gate is 7:
        



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