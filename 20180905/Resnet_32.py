import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from model_32 import *
import os
import torch.backends.cudnn as cudnn
import time
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
max_result = []

def main():
    model_dir = '../hhjung/save_Resnet32_model'
    utils.default_model_dir = model_dir
    lr = 0.1
    start_time = time.time()
    cifar10_loader()
    model = ResNet()

    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    checkpoint = utils.load_checkpoint(model_dir)
    
    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, 165):
        if epoch < 80:
            learning_rate = lr
        elif epoch < 120:
            learning_rate = lr * 0.1
        else:
            learning_rate = lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        train(model, optimizer, criterion, train_loader, epoch)
        test(model, criterion, test_loader, epoch)

        if epoch % 5 == 0:
            model_filename = 'checkpoint_%03d.pth.tar' % epoch
            utils.save_checkpoint({
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_filename, model_dir)

    utils.conv_weight_L1_printing(model.module)
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

def cifar10_loader():
    batch_size = 128
    global train_loader, test_loader
    print("Data Loading ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='../hhjung/cifar10/',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='../hhjung/cifar10/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)


def train(model, optimizer, criterion, train_loader, epoch):
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
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(model, criterion, test_loader, epoch):
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

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

    max_result.append(correct)
    utils.print_log('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%) | Max: ({})'
          .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))
    print('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}% | Max: ({}))'
          .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total, max(max_result)))


if __name__=='__main__':
    main()
