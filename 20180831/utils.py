import torch
import torch.nn as nn
import os

default_model_dir = "./"

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

def save_checkpoint(state, filename, model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
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
            print_log('percent {} {}'.format(str(child.p.data), str(child.p.data.size())))
            print('percent', child.p.data, child.p.data.size())
        elif is_leaf(child):
            continue
        else:
            routing_weight_printing(child)

def conv_weight_L1_printing(model):
    for child in model.children():
        if is_leaf(child):
            if isinstance(child, nn.Conv2d):
                print_log('{} {} {}'.format(str(child) ,str(child.weight.size()), str(child.weight.norm(1))))
                print(child, child.weight.size(), child.weight.norm(1))
        else:
            conv_weight_L1_printing(child)

def conv_weight_L2_printing(model):
    for child in model.children():
        if is_leaf(child):
            if isinstance(child, nn.Conv2d):
                print_log('{} {} {}'.format(str(child) ,str(child.weight.size()), str(child.weight.norm(2))))
                print(child, child.weight.size(), child.weight.norm(2))
        else:
            conv_weight_L1_printing(child)


def print_log(text, filename="log.txt"):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    model_filename = os.path.join(default_model_dir, filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")