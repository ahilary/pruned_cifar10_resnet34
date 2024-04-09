# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:13:50 2023

@author: Anish Hilary
"""

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from utils_atten import create_dataset_dir, view_image
import torch
import yaml

from torch.utils.data import Subset

# Load the config.yaml file
with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# Access the variables
cnf_dataset_name = config_data.get('dataset_name')
cnf_train_batch_size = config_data.get('train_batch_size')
cnf_eval_batch_size = config_data.get('eval_batch_size')


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def create_dataset():
    data_dir = create_dataset_dir(cnf_dataset_name)
    pin_memory = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
#trainloader
    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    loader_train = DataLoader(
        trainset, batch_size=cnf_train_batch_size, shuffle=True, pin_memory=pin_memory)
#testloader
    testset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    loader_test = DataLoader(
        testset, batch_size=cnf_eval_batch_size, shuffle=False, pin_memory=pin_memory)

########## subset ######
    
    tr_indices = torch.randperm(len(trainset))[:500]
    va_indices = torch.randperm(len(testset))[:500]
    
    tr_set=Subset(trainset, tr_indices)
    test_set=Subset(testset, va_indices)
    
    loader_train = DataLoader(
        tr_set, batch_size=cnf_train_batch_size, shuffle=True, pin_memory=pin_memory)
    
    loader_test = DataLoader(
        test_set, batch_size=cnf_eval_batch_size, shuffle=False, pin_memory=pin_memory)
    
################

# view the loaded images
    
    for i in range(3):
        # get some random training images
        dataiter = iter(loader_train)
        images, labels = next(dataiter)
        print(images.shape, labels)
        # show images
        view_image(torchvision.utils.make_grid(images))

        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(cnf_train_batch_size)))
    
    
    return loader_train, loader_test



