# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:25:39 2023

@author: Anish Hilary
"""

import torch
from data_creation import create_dataset
from tqdm import tqdm
from utils_atten import Averager, device
from sklearn.metrics import precision_score
import torch.nn as nn
import numpy as np


_dir = r"C:\Users\Anish Hilary\RESNET\attention_models_cifar\results\29_10-13_39_40\resnet_34/"
best_model_check = _dir + str('best_model.pth')

prune_model_dir = _dir + str('untrain_prune_model.pth')
 
un_train_model = torch.load(prune_model_dir)

best_model_check = torch.load(best_model_check)
best_model_wghts = best_model_check['model_state_dict']
un_train_model.load_state_dict(best_model_wghts)


# dataset loaders
train_loader, test_loader = create_dataset()

actual = []
prediction = []


def test(model, test_loader):
    top1 = Averager()
    model.to(device).eval()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc= "Test_Epoch", unit="batch")
        for batch_idx, batch_input in enumerate(pbar):
            images, labels = batch_input
            
            images = images.to(device)
            labels = labels.to(device)
            #print(f'labels: {labels}')
            outputs = model(images)
            #print(f'output: {outputs}')
            
            loss = criterion(outputs, labels)
           # print(f'loss: {loss}')
            o = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            accuracy = precision_score(labels, o, average='micro')
            
            actual.append(labels)
            prediction.append(o)
            
            top1.send(accuracy)
                
            running_loss += loss.item()
            pbar.set_postfix({'running_loss' : running_loss/(batch_idx+1), 'accuracy' : top1.value})
        
        # avg accuracy per epoch is compared with previous best
        accu = top1.value
    
            
        #print(round(running_loss/(batch_idx+1), 2))
        
        pbar.close()
        top1.reset
        
    return round(running_loss/(batch_idx+1), 2), accu, actual, prediction


if __name__ == '__main__':
    
    valid_loss, top1_accu, actual_targ, pred_targ = test(un_train_model, test_loader)
    act = np.array(actual_targ).reshape(-1)
    pred = np.array(pred_targ).reshape(-1)
    
    boolean_val = act == pred
    true_count = np.count_nonzero(boolean_val)
    
    accuracy = true_count / len(act)
    
    
    