import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn
from collections import OrderedDict


# one epoch means all thing learning by one learning
# residual learning image? n < 128 is it right? when did not match last thihngs is not good performance
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
start_time = time.time()
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),  # augmentation performance upgrade 7~8%
	transforms.RandomHorizontalFlip(),  # right and left reverse
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
						 # RGB MEAN when bright and dark scean are simirarly we see that what we could
						 std=(0.2471, 0.2436, 0.2616))  #
])

transform_test = transforms.Compose([
	transforms.ToTensor(),  # performance should not be change
	transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
						 # that value did not change becuase that is important, this mean that just scope is constraint in that
						 std=(0.2471, 0.2436, 0.2616))
])

# automatically download
train_dataset = datasets.CIFAR10(root='drive/app/cifar10/',
								 train=True,
								 transform=transform_train,
								 download=True)

test_dataset = datasets.CIFAR10(root='drive/app/cifar10/',
								train=False,
								transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True,
										   # data shuffle, when l oading in this batch is very good performance being generation
										   num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False,  #
										  num_workers=4)


class _FixedDenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_FixedDenseLayer, self).__init__()

        self.num_layer = 6
        self.drop_rate = drop_rate

        self.feature0 = nn.Sequential(
                nn.BatchNorm2d(num_input_features),
                nn.ReLU(),
                nn.Conv2d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.feature1 = nn.Sequential(
                nn.BatchNorm2d(num_input_features + 1 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(num_input_features + 1 * growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.feature2 = nn.Sequential(
                nn.BatchNorm2d(num_input_features + 2 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(num_input_features + 2 * growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.feature3 = nn.Sequential(
                nn.BatchNorm2d(num_input_features + 3 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(num_input_features + 3 * growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.feature4 = nn.Sequential(
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(4 * growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )
        
        self.feature_next = nn.Sequential(
                nn.BatchNorm2d(6 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(6 * growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
            )

        self.transition = nn.Sequential(
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(4 * growth_rate, growth_rate,
                         kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
        self.compress = nn.Sequential(
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate, growth_rate, kernel_size=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.compress2 = nn.Sequential(
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False),

                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(),
                nn.Conv2d(bn_size * growth_rate, growth_rate,
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
        
        
    def forward(self, x):
        
        # prob_selNet(prev_Layer, Upper_Layer_arr, percentage, training)

        f_x = self.feature0(x)
        g_x = self.feature1(torch.cat([x, f_x], 1))
        h_x = self.feature2(torch.cat([x, f_x, g_x], 1))
        i_x = self.feature3(torch.cat([x, f_x, g_x, h_x], 1))
        j_x = self.feature4(torch.cat([f_x, g_x, h_x, i_x], 1))
        k_x = self.feature4(torch.cat([g_x, h_x, i_x, j_x], 1))

        x2 = self.transition(torch.cat([h_x, i_x, j_x, k_x], 1))

        # f_x_c = self.compress2(f_x)
        g_x_c = self.compress2(g_x)
        h_x_c = self.compress2(h_x)
        i_x_c = self.compress2(i_x)
        j_x_c = self.compress2(j_x)
        k_x_c = self.compress2(k_x)

        a_x = self.feature_next(torch.cat([x2, g_x_c, h_x_c, i_x_c, j_x_c, k_x_c], 1))
        b_x = self.feature_next(torch.cat([x2, h_x_c, i_x_c, j_x_c, k_x_c, a_x], 1))
        c_x = self.feature_next(torch.cat([x2, i_x_c, j_x_c, k_x_c, a_x, b_x], 1))
        d_x = self.feature_next(torch.cat([x2, j_x_c, k_x_c, a_x, b_x, c_x], 1))
        e_x = self.feature_next(torch.cat([x2, k_x_c, a_x, b_x, c_x, d_x], 1))
        m_x = self.feature_next(torch.cat([x2, a_x, b_x, c_x, d_x, e_x], 1))
        n_x = self.feature_next(torch.cat([a_x, b_x, c_x, d_x, e_x, m_x], 1))

        return torch.cat([b_x, c_x, d_x, e_x, m_x, n_x], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()

        layer = _FixedDenseLayer(num_input_features, growth_rate, bn_size, drop_rate)
        self.add_module('fixedDenseLayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, num_init_features=16, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # Each denseblock
        num_features = num_init_features
        
        layer = _FixedDenseLayer(num_features, growth_rate, bn_size, drop_rate)
        
        self.features.add_module('fixedDenseLayer1', layer)

        num_features = 6 * growth_rate
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(3456, num_classes)

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
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=3, stride=1).view(features.size(0), -1)
        out = self.classifier(out)

        return out



model = DenseNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()

if torch.cuda.device_count() > 0:
	print("USE", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
	cudnn.benchmark = True
else:
	print("USE ONLY CPU!")

if torch.cuda.is_available():
	model.cuda()


def train(epoch):
	model.train()
	train_loss = 0 
	total = 0
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		else:
			data, target = Variable(data), Variable(target)



		optimizer.zero_grad()
		output = model(data)  # 32x32x3 -> 10*128 right? like PCA
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = torch.max(output.data, 1)
		# torch.max() : (maximum value, index of maximum value) return.
		# 1 :  row마다 max계산 (즉, row는 10개의 class를 의미)
		# 0 : column마다 max 계산
		total += target.size(0)
		correct += predicted.eq(target.data).cpu().sum()
		if batch_idx % 10 == 0:
			print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
				  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (data, target) in enumerate(test_loader):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		else:
			data, target = Variable(data), Variable(target)

		outputs = model(data)
		loss = criterion(outputs, target)

		test_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += target.size(0)
		correct += predicted.eq(target.data).cpu().sum()
	print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
		  .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(0, 165):
	if epoch < 80:
		learning_rate = learning_rate
	elif epoch < 120:
		learning_rate = learning_rate * 0.1
	else:
		learning_rate = learning_rate * 0.01
	for param_group in optimizer.param_groups:
		param_group['learning_rate'] = learning_rate

	train(epoch)
	test()  # Did not know when is it good performance.

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))