import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _Gate(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_init_features, growth_rate):
        super(_Gate, self).__init__()
        self.growth_rate = growth_rate
        self.init = num_init_features

        self.cnt = ((channels - num_init_features) // growth_rate) + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(channels//reduction, self.cnt, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        arr = []
        arr.append(x[:,:self.init,:,:]) # 0 ~ 15
        self.p_set = 1

        if self.cnt is not 1:
        
            out = self.avg_pool(x)
            out = out.permute(0, 2, 3, 1)
            out = self.relu(self.fc1(out))
            out = self.sigmoid(self.fc2(out))
            out = out.permute(0, 3, 1, 2) # batch, n+1(=num_route), 1, 1

            arr = arr + list(x[:,self.init:,:,:].split(self.growth_rate, dim=1))
        
            self.p_set = list(torch.split(out, 1, dim=1))
            p_sum = sum(self.p_set)

            for i in range(self.cnt):
                self.p_set[i] = self.p_set[i] / p_sum * self.cnt
                arr[i] = arr[i] * self.p_set[i]

        # print self.p_set
        return torch.cat(arr, 1)

class _Gate2(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_init_features, growth_rate):
        super(_Gate2, self).__init__()
        self.growth_rate = growth_rate
        self.init = num_init_features

        self.cnt = ((channels - num_init_features) // growth_rate) + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(reduction, self.cnt, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        arr = []
        arr.append(x[:,:self.init,:,:]) # 0 ~ 15
        self.p_set = 1

        if self.cnt is not 1:
        
            out = self.avg_pool(x)
            out = out.permute(0, 2, 3, 1)
            out = self.relu(self.fc1(out))
            out = self.sigmoid(self.fc2(out))
            out = out.permute(0, 3, 1, 2) # batch, n+1(=num_route), 1, 1

            print(out.size())

            arr = arr + list(x[:,self.init:,:,:].split(self.growth_rate, dim=1))
            print('arr_len', len(arr))
            print('arr_0', arr[0].size())
            print('arr_-1', arr[-1].size())
        
            self.p_set = list(torch.split(out, 1, dim=1))
            print('p_set_len', len(self.p_set))
            print('p_set_0', self.p_set[0].size())
            p_sum = sum(self.p_set)

            for i in range(self.cnt):
                self.p_set[i] = self.p_set[i] / p_sum * self.cnt
                arr[i] = arr[i] * self.p_set[i]

        # print self.p_set
        return torch.cat(arr, 1)


# class _Gate3(nn.Sequential):
#     phase = 2
#     def __init__(self, channels, reduction, num_init_features, growth_rate):
#         super(_Gate3, self).__init__()
#         self.growth_rate = growth_rate
#         self.init = num_init_features

#         self.cnt = ((channels - num_init_features) // growth_rate) + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(channels, reduction, bias=False)
#         self.relu = nn.ReLU(inplace=True)    
#         self.fc2 = nn.Linear(reduction, self.cnt, bias=False)
#         self.fc2.weight.data.fill_(0.)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         arr = []
#         arr.append(x[:,:self.init,:,:]) # 0 ~ 15
#         self.p_set = 1

#         if self.cnt is not 1:
        
#             out = self.avg_pool(x)
#             out = out.permute(0, 2, 3, 1)
#             out = self.relu(self.fc1(out))
#             out = self.sigmoid(self.fc2(out))
#             out = out.permute(0, 3, 1, 2) # batch, n+1(=num_route), 1, 1

#             arr = arr + list(x[:,self.init:,:,:].split(self.growth_rate, dim=1))
        
#             self.p_set = list(torch.split(out, 1, dim=1))
#             p_sum = sum(self.p_set)

#             for i in range(self.cnt):
#                 self.p_set[i] = self.p_set[i] / p_sum * self.cnt
#                 arr[i] = arr[i] * self.p_set[i]

#         # print self.p_set
#         return torch.cat(arr, 1)



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate, num_init_features, num_gate):
        super(_DenseLayer, self).__init__()

        if num_gate is 1:
            self.gate = _Gate(channels=num_input_features, reduction=4, num_init_features=num_init_features, growth_rate=growth_rate)    
        elif num_gate is 2:
            self.gate = _Gate2(channels=num_input_features, reduction=16, num_init_features=num_init_features, growth_rate=growth_rate)
        elif num_gate is 3:
            self.gate = _Gate2(channels=num_input_features, reduction=24, num_init_features=num_init_features, growth_rate=growth_rate)
        elif num_gate is 4:
            self.gate = _Gate2(channels=num_input_features, reduction=32, num_init_features=num_init_features, growth_rate=growth_rate)
        
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        
        out = self.gate(x)
        new_features = super(_DenseLayer, self).forward(out)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, num_init_features, num_gate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate, num_init_features, num_gate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self):
        super(_Transition, self).__init__()
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6,6,6),
                 num_init_features=16, drop_rate=0, num_classes=10, num_gate=1):

        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                growth_rate=growth_rate, drop_rate=drop_rate, num_init_features=num_init_features, num_gate=num_gate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition()
                self.features.add_module('transition%d' % (i + 1), trans)

        # Final batch norm
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

        # Linear layer
        # Official init from torch repo.

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out