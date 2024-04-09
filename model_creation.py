# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:21:23 2023

@author: Anish Hilary
"""


import torch
import os
from utils_atten import device
import torch.optim as optim
import torch.nn as nn

import resnet_34_attention 
import resnet_34_pruning
import sixth_level

import yaml
import resnet_34_pruning_6

# Load the config.yaml file
with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# Access the variables
num_classes = config_data.get('num_classes')
cnf_lr = config_data.get('lr')

'''Training done for selecting top-k best channels in each layer'''

def initial_model(train_loader, model_path, name):

    
    if name == 'resnet_34':
        
        # init_model = resnet_34_attention.resnet34_()
        # init_model.fc = nn.Linear(init_model.fc.in_features, num_classes)
        # init_model.to(device)
        
        # second_init_model = second_level.second_level_pruning()
        # second_init_model.fc = nn.Linear(second_init_model.fc.in_features, num_classes)
        # second_init_model.to(device)
        
        sixth_init_model = sixth_level.sixth_level_pruning()
        sixth_init_model.fc = nn.Linear(sixth_init_model.fc.in_features, num_classes)
        sixth_init_model.to(device)
                        
        
        
        optimizer = optim.SGD(sixth_init_model.parameters(), lr=cnf_lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cnf_lr_decay_step, gamma=0.1)
        
    # model training happens for selecting best channels
        resnet_34_pruning_6.train_for_channel_wgts(train_loader, sixth_init_model, optimizer, criterion, model_path)
        
'''Training done for selecting top-k best channels in each layer'''

def pruned_model(model_path, name, best_wgt_epoch):

    if name == 'resnet_34':
        # path of trained modules with SCA is given for class init along with epoch having optimal channel wgts
        #prune_obj = resnet_34_pruning.create_prune_model(model_path, best_wgt_epoch) 
        prune_obj = resnet_34_pruning_6.create_prune_model(model_path, best_wgt_epoch) 
        prune_obj.pruning()
        
        prune_model = torch.load(model_path.untrain_pruned_dir())

    return prune_model

if __name__ == '__main__':
    
    init = initial_model('t','m', 'resnet_34')

    pru_num = sum(p.numel() for p in init.parameters() if p.requires_grad)
    
    
    
    
    
    
    
    
    