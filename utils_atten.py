# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:26:37 2023

@author: Anish Hilary
"""


from pathlib import Path
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import numpy as np

current_dir = Path.cwd()
main_dir = current_dir.parent
data_dir = main_dir.parent


import yaml

# Load the config.yaml file
with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# Access the variables
cnf_dataset_name = config_data.get('dataset_name')
cnf_train_batch_size = config_data.get('train_batch_size')
cnf_epoch = config_data.get('epochs')


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def init_global():
    global channel_wgt_list
    channel_wgt_list = []


def create_dataset_dir(name):
    dataset_dir = current_dir.joinpath(f'{name}_dataset')
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    return dataset_dir
    
    
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)

      _, pred = output.topk(maxk, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))

      res = []
      for k in topk:
          correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
          res.append(correct_k.mul_(100.0 / batch_size))
      return res  


def start_timer(timeName="Default_timer"):
    globals()[timeName] = time.time()
    print("Timer set for '" + timeName + "'")
           
def stop_timer(timeName="Default_timer"):
    if timeName in globals():
        time_then = globals()[timeName]
        del globals()[timeName]
        secs = round(time.time() - time_then)
        print(format_time(secs, timeName))        
        return secs
    else:
        print("Toc: start time for '" + timeName + "' is not set")
        return False
    
    
def format_time(seconds,  timeName=""):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    
    if not len(timeName) == 0:
        f = "Elapsed time for '{}' is ".format(timeName)
    else:
        f = "Elapsed time is "

    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def view_image(img):
    img[0, :, :] = img[0, :, :] * 0.2023 + 0.4914    # unnormalize
    img[1, :, :] = img[1, :, :]*0.1994 + 0.4822     # unnormalize
    img[2, :, :] = img[2, :, :]*0.2010 + 0.4465     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
class checkpoints_results:
    def __init__(self, running_model, ep_ch, intermediate_run_dir = None):
        if intermediate_run_dir is None:
            dir_time = time.strftime('%d_%m-%H_%M')
            
            self.result_dir = current_dir.joinpath('results')
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
    
            self.trial_dir = self.result_dir.joinpath(f'{dir_time}_{ep_ch}')
            if not os.path.exists(self.trial_dir):
                os.makedirs(self.trial_dir)
                
            self.model_dir = self.trial_dir.joinpath(running_model)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
            self.chan_wgt_dir = self.model_dir.joinpath('channel_wghts')
            if not os.path.exists(self.chan_wgt_dir):
                os.makedirs(self.chan_wgt_dir)
                
        
        else:
            self.model_dir = Path(intermediate_run_dir)
            
            self.chan_wgt_dir = self.model_dir.joinpath('channel_wghts')
            if not os.path.exists(self.chan_wgt_dir):
                os.makedirs(self.chan_wgt_dir)

        self.best_model_accuracy = 0
            
            #latest_checkpoint = self.model_dir.joinpath('best_model.pth')
    
    def untrain_pruned_dir(self):
        return os.path.join(self.model_dir,'untrain_prune_model.pth')
        
    def save_best(self):
         return  os.path.join(self.model_dir,'best_model.pth')

    def store_chan_wgts(self, best_wt_epoch):
        return f'{self.chan_wgt_dir}/wght_mod_{best_wt_epoch}.tar'
    
    def save_latest(self):
        return os.path.join(self.model_dir, 'latest_model.pth')
        
    def epoch_plot(self, model_name, *args):
        range_list = range(1, cnf_epoch+1)
        bars = [str(num) for num in range_list]
        x_pos = np.arange(len(bars))
        
        length = int(len(args)/2)
        plot_list = args[:length]
        name_list = args[length:]
        
        plt.figure(figsize=(80,10))
        for e, all_data in enumerate(zip(plot_list,name_list)):
            graph, name = all_data
            plt.subplot(1,4,e+1)
            if type(name) is not tuple:
                if name=='mAP':
                    plt.plot(graph, label = str(name), color = 'green', linewidth=1.5)
                else:
                    plt.plot(graph, label = str(name), color = 'magenta', linewidth=1.5)
                plt.xlabel('epochs')
                plt.ylabel(str(name))
                plt.xticks(x_pos, bars)
                plt.legend()
                plt.title(f'{model_name}_model')
            else:
                color_set = ['red','blue','orange']
                for i in range(len(name)):
                    if e==0:
                        plt.plot(graph[i], label = str(name[i]), color = color_set[i], linewidth=1.5)
                    else:
                        plt.plot(graph[i], label = str(name[i]), linewidth=1.5)
                    plt.xlabel('epochs')
                    plt.ylabel(str(name[0])[-4:])
                    plt.xticks(x_pos, bars)
                    plt.legend()
                    plt.title(f'{model_name}_model')
        plt.savefig(os.path.join(self.model_dir,'plots.pdf'))
        
        
        