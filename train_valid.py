# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:42:19 2023

@author: Anish Hilary
"""
                                                                                                                                                      
from sklearn.metrics import precision_score
from utils_atten import Averager, device
import torch
from tqdm import tqdm


def train(train_loader, model, optimizer, criterion, epoch):
    
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Train_Epoch_{epoch}", unit="batch")
    for batch_idx, batch_input in enumerate(pbar):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = batch_input
        
        images = images.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
# model       
        outputs = model(images)

# loss
        loss = criterion(outputs, labels)
        
# optimizers
        loss.backward()
        optimizer.step()

        del images, labels, outputs
        torch.cuda.empty_cache()
        
        # print statistics
        running_loss += loss.item()
        # if e % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {e + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
            
        pbar.set_postfix({'running_loss' : round(running_loss/(batch_idx+1), 2)})
        
    pbar.close()
     
        
    return round(running_loss/(batch_idx+1), 2), model




#%%


#%%
'''Validation'''

def test(test_loader, model, criterion, epoch):
    top1 = Averager()
 
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Test_Epoch_{epoch}", unit="batch")
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

            top1.send(accuracy)
                
            running_loss += loss.item()
            pbar.set_postfix({'running_loss' : running_loss/(batch_idx+1), 'accuracy' : top1.value})
        
        # avg accuracy per epoch is compared with previous best
        accu = top1.value

            
        #print(round(running_loss/(batch_idx+1), 2))
        
        pbar.close()
        top1.reset
        
    return round(running_loss/(batch_idx+1), 2), accu

  # top5 = AverageMeter()
            # prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            # top1.update(prec1[0], images.size(0))
            # top5.update(prec5[0], images.size(0))
 #                     'scheduler':scheduler.state_dict(),  
