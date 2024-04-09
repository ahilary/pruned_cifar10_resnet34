# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:46:53 2023

@author: Anish Hilary
"""


import os

from utils_atten import device, checkpoints_results, start_timer, stop_timer
from torchvision.models import resnet34
from data_creation import create_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from train_valid import train, test
import model_creation


# Load the config.yaml file
import yaml

with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)


# Access the variables
cnf_epoch = config_data.get('epochs')
cnf_lr = config_data.get('lr')
cnf_lr_decay_step = config_data.get('lr_decay_step')
cnf_dataset_name = config_data.get('dataset_name')
num_classes = config_data.get('num_classes')


print('Model is running in {}'.format(device))

# start training from intermediate point
inter_dir = None

# init to store results and checkpoints
model_saver = checkpoints_results('resnet_34', cnf_epoch, inter_dir)

# dataset loaders
train_loader, test_loader = create_dataset()


# If Training is interupted
if inter_dir and os.path.isfile(model_saver.untrain_pruned_dir()):
    
    if os.path.isfile(model_saver.save_latest()):
        checkpoint = torch.load(model_saver.save_latest())
        pruned_model = torch.load(model_saver.untrain_pruned_dir())
        pruned_model.to(device)
        start_epoch = checkpoint['epoch']+1
        
    else:
        pruned_model = torch.load(model_saver.untrain_pruned_dir())
        pruned_model.to(device)
        start_epoch = 1

if inter_dir and not os.path.isfile(model_saver.untrain_pruned_dir()):
    # prune less important channels
    pruned_model = model_creation.pruned_model(model_saver, 'resnet_34', best_wgt_epoch=60)
    pruned_model.to(device)
    
    start_epoch = 1
    
if inter_dir is None:
    
    # collect channel weights
    model_creation.initial_model(train_loader, model_saver, 'resnet_34')

    # prune less important channels
    pruned_model = model_creation.pruned_model(model_saver, 'resnet_34', best_wgt_epoch=60)
    pruned_model.to(device)
    
    start_epoch = 1



optimizer = optim.SGD(pruned_model.parameters(), lr=cnf_lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

train_epoch_loss = []
valid_epoch_loss = []

train_epoch_time = []
valid_epoch_time = []

valid_epoch_accuracy = []
    
    
# normal resnet model
normal_model = resnet34()
norm_params = sum(p.numel() for p in normal_model.parameters() if p.requires_grad)
print(f'The learnable parameters before pruning: {norm_params}')

# pruned resnet model
learnable_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
print(f'The learnable parameters after pruning: {learnable_params}')

#best_model_accuracy = 0

for epoch in range(start_epoch, cnf_epoch + 1):
    if not os.path.isfile(model_saver.save_latest()):
        pass
        
    else:
        checkpoint = torch.load(model_saver.save_latest())
        pruned_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']+1
        train_epoch_time = checkpoint['train_epoch_time']
        train_epoch_loss = checkpoint['train_epoch_loss']
        valid_epoch_loss = checkpoint['valid_epoch_loss']
        valid_epoch_time = checkpoint['valid_epoch_time']
        valid_epoch_accuracy = checkpoint['valid_epoch_accuracy']
        model_saver.best_model_accuracy = checkpoint['best_accuracy']
    
# Training
    start_timer('Training')  
    train_loss, pruned_model = train(train_loader, pruned_model, optimizer, criterion, epoch)
    train_epoch_time.append(stop_timer('Training'))
    
    train_epoch_loss.append(train_loss)

# Validation
    start_timer('Validation')  
    valid_loss, top1_accu = test(test_loader, pruned_model, criterion, epoch)
    valid_epoch_time.append(stop_timer('Validation'))
    
    valid_epoch_loss.append(valid_loss)
    valid_epoch_accuracy.append(top1_accu)

    
    #scheduler.step()

# save best model
    if top1_accu > model_saver.best_model_accuracy:
        model_saver.best_model_accuracy = top1_accu
        torch.save({
                'epoch': epoch,
                'model_state_dict': pruned_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_epoch_time': train_epoch_time,
                'train_epoch_loss': train_epoch_loss,
                'valid_epoch_loss': valid_epoch_loss,
                'valid_epoch_time': valid_epoch_time,
                'valid_epoch_accuracy': valid_epoch_accuracy,
                'learnable_params': learnable_params,
                'best_accuracy': model_saver.best_model_accuracy
                }, model_saver.save_best())
        
        
# save checkpoint
    torch.save({
            'epoch': epoch,
            'model_state_dict': pruned_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_epoch_time': train_epoch_time,
            'train_epoch_loss': train_epoch_loss,
            'valid_epoch_loss': valid_epoch_loss,
            'valid_epoch_time': valid_epoch_time,
            'valid_epoch_accuracy': valid_epoch_accuracy,
            'learnable_params': learnable_params,
            'last_accuracy': top1_accu,
            'best_accuracy': model_saver.best_model_accuracy
            }, model_saver.save_latest())
            
    



# Model plots
model_saver.epoch_plot(cnf_dataset_name, (train_epoch_loss, valid_epoch_loss),
                       (valid_epoch_accuracy),
                       (train_epoch_time),
                       (valid_epoch_time),
                       ('train_loss','valid_loss'),
                       ('valid_accuracy'),
                       ('train_epoch_time'),
                       ('valid_epoch_time'))



