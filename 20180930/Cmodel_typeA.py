import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None, init_block=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.init_block = init_block
        
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
            out = sum(x) # change to weighted sum
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
    def __init__(self, num_classes=10, resnet_layer=56):
        super(ResNet, self).__init__()

        self.features = nn.Sequential()
        num_features = 16

        self.features.add_module('conv1', nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('bn1', nn.BatchNorm2d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        if resnet_layer is 14:
            block_config = (2,2,2)
        elif resnet_layer is 20:
            block_config = (3,3,3)
        elif resnet_layer is 32:
            block_config = (5,5,5)
        elif resnet_layer is 44:
            block_config = (7,7,7)
        elif resnet_layer is 56:
            block_config = (9,9,9)
        elif resnet_layer is 110:
            block_config = (18,18,18)
            

        
        for i, num_layers in enumerate(block_config):
            if i is 0:
                layer = nn.Sequential()
                layer.add_module('layer%d_0' % (i+1), BasicBlock(in_channels=num_features
                        , out_channels=num_features, stride=1, downsample=None, init_block=True))
            else:
                layer = nn.Sequential()
                layer.add_module('layer%d_0' % (i+1), BasicBlock(in_channels=num_features
                        , out_channels=num_features*2, stride=2, downsample=True, init_block=False))
                num_features = num_features * 2

            for j in range(1, num_layers):
                layer.add_module('layer%d_%d' % (i+1, j), BasicBlock(in_channels=num_features
                        , out_channels=num_features, stride=1, downsample=None, init_block=False))

            self.features.add_module('layer%d' % (i + 1), layer)
            # if i != len(block_config) - 1:
            #     self.features.add_module('transition%d' % (i + 1), _Transition(in_channels=num_features, out_channels=num_features * 2))
            #     num_features = num_features * 2

        # Final batch norm

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = sum(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out