import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from model import *
import os
import torch.backends.cudnn as cudnn
import time
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def main(model_dir, model, dataset):
    utils.default_model_dir = model_dir
    # model = model
    lr = 0.1
    start_time = time.time()

    if dataset == 'cifar10':
        train_loader, test_loader = utils.cifar10_loader()
    elif dataset == 'cifar100':
        train_loader, test_loader = utils.cifar100_loader()
    

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
            utils.print_log('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
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


layer_set = [14, 20, 32, 44, 56, 110]

def do_learning(model_dir, db, layer):
    max_result = []
    model_selection = ResNet(num_classes=db, resnet_layer=layer)
    dataset = 'cifar' + str(db)
    main(model_dir, model_selection, dataset)

if __name__=='__main__':
    
    for i in range(10):
        model_dir = '../hhjung/Basemodel/cifar10_2/Resnet110_' + str(i)
        do_learning(model_dir, 10, layer_set[5])

        model_dir = '../hhjung/Basemodel/cifar100_2/Resnet110_' + str(i)
        do_learning(model_dir, 100, layer_set[5])
    