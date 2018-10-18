import torch
import torch.nn as nn
from torchvision import datasets, transforms
import csv
import os
from operator import add

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'

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
                # print('False', model)
            else:
                model.weight.requires_grad = True
                # print('True', model)
        return
    
    for child in model.children():

        if is_leaf(child):
            if hasattr(child, 'weight'):
                if child.weight.requires_grad:
                    child.weight.requires_grad = False
                    # print('False', child)
                else:
                    child.weight.requires_grad = True
                    # print('True', child)
        else:
            switching_learning(child)


def routing_weight_printing(model):
    for child in model.children():
        if hasattr(child, 'phase'):
            print_log('percent {} {}'.format(str(child.p.data), str(child.p.data.size())))
            print('percent', child.p.data, child.p.data.size())
        elif is_leaf(child):
            continue
        else:
            routing_weight_printing(child)


def weight_extract(model):
    global c
    if c is not None:
        for child in model.children():
            if hasattr(child, 'phase'):
                c = torch.cat([c, child.p.view(-1,1)], 1)
            elif is_leaf(child):
                continue
            else:
                weight_extract(child)


def save_to_csv():
    global str_w
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)

    model_filename = os.path.join(default_model_dir, csv_file_name)
    with open(model_filename, 'a') as fout:
        fout.write(str_w)

def del_csv_weight_for_test():
    model_filename = os.path.join(default_model_dir, csv_file_name)
    if os.path.exists(model_filename):
        os.remove(model_filename)
        print('delete ', csv_file_name)