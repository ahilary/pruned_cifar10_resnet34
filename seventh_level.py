# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:57:37 2023

@author: Anish Hilary
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:56:46 2023

@author: Anish Hilary
"""


import torch.nn as nn
import resnet_34_attention 
from torch.nn.parameter import Parameter
import torch


def seventh_level_pruning():
    
    base_second_prune = resnet_34_attention.resnet34_()
    
    base_second_prune.fc = nn.Linear(base_second_prune.fc.in_features, 10)
    
    
    # load the state dict of best epoch into the pruned model
    
    direct = r'C:\Users\Anish Hilary\RESNET\attention_models_cifar\results\26_10-16_02_40\resnet_34/'
    bestmoddir = direct + str('best_model.pth') 
    
    check = torch.load(bestmoddir)
    statedict = check['model_state_dict']
    
    untrainprunedir = direct + str('untrain_prune_model.pth')
    train_pruned_model = torch.load(untrainprunedir)
    
    train_pruned_model.load_state_dict(statedict)
    
    
    # now must transfer the weights from train_pruned_model to base_second_prune (which has SCA module)
    
    conv_out = 0
    
    for base in base_second_prune.named_modules():
        base_name, base_mod = base
        
        for prune in train_pruned_model.named_modules():
            prune_name, prune_mod = prune
            
            if isinstance(base_mod and prune_mod, nn.Conv2d):
                # print(base_name,base_mod)
                # print(prune_name, prune_mod)
                
                if base_name == prune_name:
                    reduced_shape = prune_mod.weight.shape
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:reduced_shape[0], :reduced_shape[1], :, :].clone())
                        base_mod.weight = Parameter(prune_mod.weight.clone())
                        
                    base_mod.in_channels = reduced_shape[1]
                    base_mod.out_channels = reduced_shape[0]
                    
                    conv_out = reduced_shape[0]
                    
                    #print(base_mod.weight.shape)
            
            if isinstance(base_mod and prune_mod, nn.BatchNorm2d):
                # print(base_name,base_mod)
                # print(prune_name, prune_mod)
                
                if base_name == prune_name:
                    reduced_weight_shape = prune_mod.weight.shape
                
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:reduced_weight_shape[0]].clone())
                        base_mod.bias = Parameter(base_mod.bias[:reduced_weight_shape[0]].clone())  
                        base_mod.running_mean = base_mod.running_mean[:reduced_weight_shape[0]].clone()
                        base_mod.running_var = base_mod.running_var[:reduced_weight_shape[0]].clone()
                        
                        
                        base_mod.weight = Parameter(prune_mod.weight.clone())
                        base_mod.bias = Parameter(prune_mod.bias.clone())  
                        base_mod.running_mean = prune_mod.running_mean.clone()
                        base_mod.running_var = prune_mod.running_var.clone()
                        
                    base_mod.num_features = reduced_weight_shape[0]
                   
                    
            if isinstance(base_mod and prune_mod, nn.Linear):
                # print(base_name,base_mod)
                # print(prune_name, prune_mod)
                
                if base_name == prune_name:
                    linear_shape = prune_mod.weight.shape
                    
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:, linear_shape[1]].clone())
                        base_mod.weight = Parameter(prune_mod.weight.clone())
                        
                        base_mod.bias = Parameter(prune_mod.bias.clone())
                    
                    base_mod.in_features = linear_shape[1]
            
            if 'channel_att.gn' in base_name:
                with torch.no_grad():
                    base_mod.weight = Parameter(base_mod.weight[:, :conv_out, :, :].clone())
                    base_mod.bias = Parameter(base_mod.bias[:, :conv_out, :, :].clone())

                    
    return base_second_prune


def get_prune_base():
    
    direct = r'C:\Users\Anish Hilary\RESNET\attention_models_cifar\results\26_10-16_02_40\resnet_34/'
    bestmoddir = direct + str('best_model.pth') 
    
    check = torch.load(bestmoddir)
    statedict = check['model_state_dict']
    
    untrainprunedir = direct + str('untrain_prune_model.pth')
    train_pruned_model = torch.load(untrainprunedir)
    
    train_pruned_model.load_state_dict(statedict)
    
    return train_pruned_model


if __name__ == '__main__':
    seventh_base_model = seventh_level_pruning()
    