# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:49:49 2023

@author: Anish Hilary
"""


from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
import  torch.nn as nn
from utils_atten import start_timer, stop_timer, device
from torch.nn.parameter import Parameter

#from dataset import dataset_creator
#import torch.optim as optim
import pickle
import os
#from torch.utils.data import DataLoader
import resnet_34_attention 
import resnet_34_normal
import fourth_level
import torch.nn.utils as utils

#from torchvision.models import resnet34

def train_for_channel_wgts(train_loader, model, optimizer, criterion, model_path):
    
    # list of dict --> [{0:[...]},{1:[...]},{2:[...]}.....{0:[...]},{1:[...]},{2:[...]}...]
    # len(list) => num_layer * num_batches * num_epochs
    
    epoch_level_weights = []
    for epoch in range(15):
        running_loss = 0.0
        epoch_weights_list = []
        
        channel_weights_list = []
        #res_mod = resnet34().to(device)
        pbar = tqdm(train_loader, desc=f"Training_wght_{epoch}", unit="batch")
        for e, batch_input in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = batch_input
            
            images = images.to(device)
            labels = labels.to(device)
            
            # collects channel weights of each layer for each batch
            batch_wgt = {}
                    
            # zero the parameter gradients
            optimizer.zero_grad()
    # model       
            #outuu = res_mod(images)
            outputs = model(images)
            
            
            # check = int((outputs != outputs).sum())
            # if(check>0):
            #     print("your data contains Nan")
            # else:
            #     print("Your data does not contain Nan, it might be other problem")

    # loss
            loss = criterion(outputs, labels)
            
            
    # optimizers
            loss.backward()
            optimizer.step()

            del images, labels, outputs
            torch.cuda.empty_cache()
            
            running_loss += loss.item()
                
        # load pkl, each layer weights of a batch is appended    
        # list of lists [[layer_0_channel_weight],[layer_1_channel_weight],[layer_2_channel_weight]...]
            with open('wght_list.pkl', 'rb') as f:
                wgt = pickle.load(f)
                for i in range(len(wgt)):
                    batch_wgt[i] = np.array(wgt[i])
                   
                channel_weights_list.append(batch_wgt)
                epoch_weights_list.append(batch_wgt)

                
            pbar.set_postfix({'running_loss' : round(running_loss/(e+1), 2)})
            
            #print(round(running_loss/(e+1), 2))
        pbar.close()
            
        each_epoch_wght = channel_weight_compile(epoch_weights_list)
        epoch_level_weights.append(each_epoch_wght)
        

        
        with open(f'{model_path.chan_wgt_dir}/wght_list_{epoch}.pkl', 'wb') as f:
            pickle.dump(each_epoch_wght, f)
        
        torch.save({'weight_model': model, 'channel_weights_list':channel_weights_list,
                    'epoch_level_weights':epoch_level_weights}, f'{model_path.chan_wgt_dir}/wght_model.tar')
    
    #torch.save({'weight_model': model, 'channel_weights_list':channel_weights_list}, 'wgt_mod.tar')  
    
    del model, channel_weights_list, batch_wgt




def channel_weight_compile(channel_weights_list):
    # chan_wgt = [{0:[[],[],[],...]}, {1:[[],[],[],...]}, {2:[[],[],[],...]}]
    chan_wgt = defaultdict(list)
    for d in channel_weights_list: 
        for key, value in d.items():
            chan_wgt[key].append(value)
    
    # layerwise_channel_weight = {0:[ , , , ], 1:[ , , , ], 2:[ , , , ]}
    layerwise_channel_weight = {}
    for layer in range(len(chan_wgt)):
        layerwise_channel_weight[layer] = np.mean(np.stack(chan_wgt[layer]), axis=0)
    
    del chan_wgt, channel_weights_list
    
    return layerwise_channel_weight


class create_prune_model:
    
    def __init__(self, model_path, best_wt_epoch, existing=None):
        
        self.model_path = model_path
        if existing:
            check_trained_model = torch.load(self.model_path.store_chan_wgts(best_wt_epoch, existing))
        else:
            check_trained_model = torch.load(f'{model_path.chan_wgt_dir}/wght_model.tar')
        
        ######### TEMPORARY
        #check_trained_model = torch.load('wgt_mod.tar')
        #########
        self.trained_model = check_trained_model['weight_model']
        
        #learnable_params = sum(p.numel() for p in self.trained_model.parameters() if p.requires_grad)
        #print(f'The learnable parameters Before Pruning: {learnable_params}')
        
        self.channel_wgts_layerwise = self.channel_weights(check_trained_model['channel_weights_list'])
        
  # if we iterate through the dict we cannot del it so converting keys to list
        self.dict_trained = self.trained_model.state_dict()
        self.trained_list = list(self.dict_trained)
        for param_names in self.trained_list:
            if 'channel_att'  in param_names :
                del self.dict_trained[param_names]
            if 'spatial_att'  in param_names :
                del self.dict_trained[param_names]
            if 'lay_4_wghts'  in param_names :
                del self.dict_trained[param_names]
  # list length reduced      
        self.trained_list = list(self.dict_trained)        
  # new init model contains un-trained weights    
        self.untrain_prune_model = fourth_level.get_prune_base()
        self.untrained_list = list(self.untrain_prune_model.state_dict())
        
  # check if the added modules are properly removed
        assert list(self.trained_list) == list(self.untrained_list), 'Layers in trained and un-trained model should be same'
        
  # moving the weights to the untrained model before pruning

        try:
            self.untrain_prune_model.load_state_dict(self.dict_trained)
        except:
            print('The trained and untrained module shapes are not equal')
        else:
            print('The trained and untrained module shapes are equal')
        
        
        del self.trained_model, self.trained_list, self.untrained_list
        
        
        self.conv1_idx_in = None
    
    
    def channel_weights(self, channel_weights_list):
        # chan_wgt = [{0:[[],[],[],...]}, {1:[[],[],[],...]}, {2:[[],[],[],...]}]
        chan_wgt = defaultdict(list)
        for d in channel_weights_list: 
            for key, value in d.items():
                chan_wgt[key].append(value)
        
        # layerwise_channel_weight = {0:[ , , , ], 1:[ , , , ], 2:[ , , , ]}
        layerwise_channel_weight = {}
        for layer in range(len(chan_wgt)):
            layerwise_channel_weight[layer] = np.mean(np.stack(chan_wgt[layer]), axis=0)
        
        del chan_wgt, channel_weights_list
        
        return layerwise_channel_weight
            
    
    def mask_channels_on_probs(self):
        # layer threshold after which the percentage of pruning is changed
        prune_prob = {0 : 0.1, 1 : 0.1, 2 : 0.1, 3 : 0.1}
        layer_id = 1
        layer_channels_in_binary= []
        conv = 1
        num_channel_layer = []
        
        for n,m in self.untrain_prune_model.named_modules():
            if isinstance (m,nn.Conv2d) and n.startswith != 'fc':
                if layer_id <= 6:
                        stage = 0
                elif layer_id <= 15:
                        stage = 1
                elif layer_id <= 28:
                        stage = 2
                else:
                        stage = 3
                if conv==1:
                    conv+=1
                    continue
                elif conv!=1:
                    prune_prob_stage = prune_prob[stage]
                    out_channels = m.weight.data.shape[0]
                    # channel weight list of 'layer_id+1'th layer, cz first layer not considered
                    channel_weight = self.channel_wgts_layerwise[layer_id-1]
                    # 10% of the outchannels are pruned, if prune_prob_stage == 0.1
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    if not num_keep%8 == 0:
                        num_keep = num_keep-(num_keep%8)
                    # asc_sort the weights and give the index
                    arg_max = np.argsort(channel_weight)
                    # reverse_sort the weights and remove extra channels
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = torch.zeros(out_channels)
                    # channel idx with value 1 are to be kept
                    mask[arg_max_rev.tolist()] = 1
                    layer_channels_in_binary.append(mask)
                    num_channel_layer.append(num_keep)
                    layer_id += 1
                    continue
                layer_id += 1
        
        del prune_prob_stage, out_channels, channel_weight, num_keep, arg_max, arg_max_rev, mask
        
        return layer_channels_in_binary, num_channel_layer
            
    def pruning(self):
        
        layer_channels_in_binary, num_channel_layer = self.mask_channels_on_probs()

        layer_id_in_model = 0
        conv_count = 1
        for base in self.untrain_prune_model.named_modules():
            base_name, base_mod = base

            
            if isinstance(base_mod, nn.Conv2d):
                # print(layer_id_in_model)
                # print(base_name, base_mod)
                
                if conv_count == 1:
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight.clone())
                    
                
                elif conv_count == 2:
                    self.conv1_idx_in = np.arange(64)
                    mask = layer_channels_in_binary[layer_id_in_model]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask)))

                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[idx_out.tolist(), :, :, :].clone())
                    
                    base_mod.out_channels = len(idx_out)
                    

                elif conv_count>2 and 'conv1' in base_name:
                    mask_in = layer_channels_in_binary[layer_id_in_model-1]
                    self.conv1_idx_in = np.squeeze(np.argwhere(np.asarray(mask_in)))
                        
                    mask_out = layer_channels_in_binary[layer_id_in_model]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out)))
                    
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:, self.conv1_idx_in.tolist(), :, :].clone())
                        base_mod.weight = Parameter(base_mod.weight[idx_out.tolist(), :, :, :].clone())
                    
                    base_mod.in_channels = len(self.conv1_idx_in)
                    base_mod.out_channels = len(idx_out)
                    
                
                elif conv_count>2 and 'conv2' in base_name:
                    mask_in = layer_channels_in_binary[layer_id_in_model-1]
                    idx_in = np.squeeze(np.argwhere(np.asarray(mask_in)))
                        
                    mask_out = layer_channels_in_binary[layer_id_in_model]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out)))

                    
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:, idx_in.tolist(), :, :].clone())
                        base_mod.weight = Parameter(base_mod.weight[idx_out.tolist(), :, :, :].clone())
                    
                    base_mod.in_channels = len(idx_in)
                    base_mod.out_channels = len(idx_out)
                    

                elif conv_count>2 and 'conv_down' in base_name:
                    
                    # print(layer_id_in_model)
                    # print(base_name, base_mod)
                        
                    mask_out = layer_channels_in_binary[layer_id_in_model]
                    idx_out = np.squeeze(np.argwhere(np.asarray(mask_out)))
                    
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[:, self.conv1_idx_in.tolist(), :, :].clone())
                        base_mod.weight = Parameter(base_mod.weight[idx_out.tolist(), :, :, :].clone())
                    
                    base_mod.in_channels = len(self.conv1_idx_in)
                    base_mod.out_channels = len(idx_out)
                    
                    
    
    
            elif isinstance(base_mod, nn.BatchNorm2d):
                # print(layer_id_in_model)
                # print(base_name, base_mod)
                
                if conv_count == 1:
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight.clone())
                        base_mod.bias = Parameter(base_mod.bias.clone())  
                        base_mod.running_mean = base_mod.running_mean.clone()
                        base_mod.running_var = base_mod.running_var.clone()
                    conv_count += 1
                    
                
                elif conv_count > 1:
                    mask = layer_channels_in_binary[layer_id_in_model]
                    idx = np.squeeze(np.argwhere(np.asarray(mask)))
                        
                    with torch.no_grad():
                        base_mod.weight = Parameter(base_mod.weight[idx.tolist()].clone())
                        base_mod.bias = Parameter(base_mod.bias[idx.tolist()].clone())  
                        base_mod.running_mean = base_mod.running_mean[idx.tolist()].clone()
                        base_mod.running_var = base_mod.running_var[idx.tolist()].clone()
                    

                    layer_id_in_model += 1
                    conv_count += 1
                    
                    base_mod.num_features = len(idx)

                        
                
            elif isinstance(base_mod, nn.Linear):
                idx = [i for i in range(320)]
                base_mod.weight = Parameter(base_mod.weight[:, idx].clone())
                base_mod.bias = Parameter(base_mod.bias.clone())
                
                base_mod.in_features = len(idx)
    
        torch.save(self.untrain_prune_model, self.model_path.untrain_pruned_dir())
        #torch.save(self.untrain_prune_model, 'pruned_model.pth')
        
        return self.untrain_prune_model


if __name__ == '__main__':

# prob_model
    backbone = resnet_34_attention.resnet34_()

    
    # train_for_channel_wgts(train_loader, model, optimizer)
    
    prune_obj = create_prune_model('r','t')
    prunemodel = prune_obj.pruning()
    

    # for name,params in prunemodel.named_parameters():
    #     print(name,':',params.size())
    # print(prunemodel)
       
    
    imp = torch.randn(2,3,128,128)
    out = prunemodel(imp)
    
    pru_num = sum(p.numel() for p in prunemodel.parameters() if p.requires_grad)
    mod_num = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    
    print(f'pruned: {pru_num}')
    print(f'not-pruned: {mod_num}')
    

        
    
    
    

        
        
        
        
        
    

           

        
    
    
    