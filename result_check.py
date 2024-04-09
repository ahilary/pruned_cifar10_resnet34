# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:48:23 2023

@author: Anish Hilary
"""


import torch

result_dir = r'C:\Users\Anish Hilary\RESNET\attention_models_cifar\results\29_10-13_39_40\resnet_34/latest_model.pth'

result_dict = torch.load(result_dir)

print(f"The total epochs : {result_dict['epoch']}")
print(f"Best accuracy : {max(result_dict['valid_epoch_accuracy'])}")
#print(f"Best accuracy : {result_dict['best_accuracy']}")
print(f"Parameters : {result_dict['learnable_params']}")


