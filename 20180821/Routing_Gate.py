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

p_decay = torch.tensor([], device='cuda:0')
p_decay_rate = 0.1
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


def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def init_learning(model):
    for child in model.children():
        if hasattr(child, 'phase'):
            turn_off_learning(child)
        elif is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = True
                # print('True', child)
        else:
            init_learning(child)

def init_learning_phase4(model):
    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = True
                # print('True', child)
        else:
            init_learning_phase4(child)

def turn_off_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            model.weight.requires_grad = False
            # print('False', model)
        return

    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = False
                # print('False', child)
        else:
            turn_off_learning(child)


def switching_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            if model.weight.requires_grad:
                model.weight.requires_grad = False
                print('False', model)
            else:
                model.weight.requires_grad = True
                print('True', model)
        return
    
    for child in model.children():

        if is_leaf(child):
            if hasattr(child, 'weight'):
                if child.weight.requires_grad:
                    child.weight.requires_grad = False
                    print('False', child)
                else:
                    child.weight.requires_grad = True
                    print('True', child)
        else:
            switching_learning(child)



class _Gate(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, out_num):
        super(_Gate, self).__init__()
        self.out_num = out_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.tanh = nn.Tanh()        
        self.fc2 = nn.Linear(channels // reduction, out_num, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.softmax = nn.Softmax(3)

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.tanh(self.fc1(out))
        out = self.softmax(self.fc2(out))
        out = out.permute(0, 3, 1, 2)
        # print('1',out.size())
        out = torch.sum(torch.t(out.view(-1,self.out_num)),1) # avg p
        # print('2',out.size())
        out = out / torch.sum(out)

        self.p = out

        return x * out[0] + res * out[1]


# class _Gate(nn.Sequential):
#     phase = 2
#     def __init__(self, channels, reduction, out_num, batch_num = 128):
#         super(_Gate, self).__init__()

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         # self.one = torch.tensor([1.], requires_grad=False, device='cuda:0')
#         self.fc1 = nn.Linear(2 * channels * batch_num, channels * batch_num // reduction, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(channels * batch_num // reduction, out_num * batch_num // reduction, bias=False)
#         self.tanh = nn.Tanh()
#         self.fc3 = nn.Linear(out_num * batch_num // reduction, out_num, bias=False)
#         self.fc3.weight.data.fill_(0.)
#         self.softmax = nn.Softmax()

#     def forward(self, x, res):
#         x_ = self.avg_pool(x)
#         res_ = self.avg_pool(res)
#         out = torch.cat([x_,res_], 1)        
#         out = out.view(-1)
#         out = self.relu(self.fc1(out))
#         out = self.tanh(self.fc2(out))
#         out = self.softmax(self.fc3(out))
#         self.p = out

#         return x * out[0] + res * out[1]


# class _Gate2(nn.Sequential):
#     phase = 2
#     def __init__(self, channels, reduction, out_num, batch_num = 128):
#         super(_Gate2, self).__init__()
#
#         self.conv_x = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
#         self.conv_res = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
#
#     def forward(self, x, res):
#         x_ = self.conv_x(x) # batch, 1, W, H
#         res_ = self.conv_res(res) # batch, 1, W, H
#         out = torch.cat([x_,res_], 1)        
#
#         return x * out[0] + res * out[1]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_num=2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.gate = _Gate(channels=self.expansion*planes, reduction=4, out_num=out_num)
        # self.gate = _Gate(channels=self.expansion*planes, reduction=4,  out_num=out_num)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        global p_decay, is_state
        out = F.relu(self.bn1(self.conv1(x)))
        residual = self.shortcut(x)
        out = self.gate(out, residual)
        
        # if is_state.check_is_phase2() and not is_state.check_is_test():
        #     p_decay = torch.cat([p_decay,p],0)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


model = ResNet(BasicBlock, [2, 2, 2, 2])
optimizer = optim.SGD(model.parameters(), learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)
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
    global p_decay, is_state
    model.train()
    train_loss = 0 
    total = 0
    correct = 0
    train_loss2 = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
            data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)  # 32x32x3 -> 10*128 right? like PCA
        loss = criterion(output, target)
        
        # if is_state.check_is_phase2():
        #     loss2 = p_decay.size()[0] - p_decay.pow(2).sum()
        #     loss2 = loss2 * p_decay_rate

        #     train_loss2 += loss2
        #     loss = loss + loss2
        #     p_decay = torch.tensor([], device='cuda:0')
        
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
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) |  Loss2: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), train_loss2 / (batch_idx + 1), 100. * correct / total, correct, total))


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
    

class is_on(object):
    def __init__(self):
        self.is_phase2 = False
        self.is_test = False
        self.is_phase4 = False

    def check_is_phase2(self):
        return self.is_phase2

    def change_phase2(self):
        if self.is_phase2:
            self.is_phase2 = False
        else:
            self.is_phase2 = True

    def change_on_phase2(self):
        if not self.is_phase2:
            self.is_phase2 = True

    def check_is_phase4(self):
        return self.is_phase4

    def change_on_phase4(self):
        if not self.is_phase4:
            self.is_phase4 = True

    def check_is_test(self):
        return self.is_test

    def change_test(self):
        if self.is_test:
            self.is_test = False
        else:
            self.is_test = True

def save_checkpoint(state, filename):

    model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint():

    model_dir = 'drive/app/torch/save_Routing_Gate_2'
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)

    return state

def routing_weight_printing(model):
    for child in model.children():
        if hasattr(child, 'phase'):
            print('percent', child.p.data)
        elif is_leaf(child):
            continue
        else:
            routing_weight_printing(child)

def conv_weight_L1_printing(model):
    for child in model.children():
        if is_leaf(child):
            if isinstance(child, nn.Conv2d):
                print(child, child.weight.size(), child.weight.norm(1))
        else:
            conv_weight_L1_printing(child)


init_learning(model.module)
# print(model)

is_state = is_on()

start_epoch = 0
checkpoint = load_checkpoint()
if not checkpoint:
    pass
else:
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, 240):

    if epoch == 200:
        is_state.change_on_phase4()
        is_state.change_on_phase2()
        init_learning_phase4(model.module)
        # p_decay_rate = p_decay_rate * 0.5

    if epoch < 120:
        l_r = learning_rate
    elif epoch < 160:
        l_r = learning_rate * 0.1
    else:
        l_r = learning_rate * 0.01

    # if is_state.check_is_phase2:
    #     l_r = l_r * 0.1

    for param_group in optimizer.param_groups:
        param_group['learning_rate'] = l_r

    train(epoch)

    if epoch % 5 == 0:
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_filename)

    # is_state.change_test()
    test()  
    # is_state.change_test()

    routing_weight_printing(model.module)

    if epoch % 5 == 4 and not is_state.check_is_phase4():
        is_state.change_phase2()
        switching_learning(model.module)

    if epoch % 10 == 0:
        conv_weight_L1_printing((model.module))

# routing_weight_printing(model.module)
conv_weight_L1_printing((model.module))

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))