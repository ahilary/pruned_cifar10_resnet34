# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:12:18 2023

@author: Anish Hilary
"""

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64, eps = 1e-5):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.compress = ChannelPool()
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()
        self.eps = eps

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn= self.compress(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels,  pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.gn = GroupNorm(gate_channels)
        self.pool_types = pool_types
        self.channel_prob = []
    def forward(self, x):

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn( avg_pool )

            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn( max_pool )

            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gn(lp_pool)
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.gn(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid( channel_att_sum ).expand_as(x)
        
        scale1 = F.sigmoid(channel_att_sum).unsqueeze(1)
        scale2 = scale1.squeeze(3)
        scale3 = scale2.squeeze(3)
        scale4 = scale3.detach().cpu()
        
        channel_wt = torch.mean(scale4,dim=0).numpy().squeeze().tolist()
        
        
        return x * scale, channel_wt

class SCA(nn.Module):
    def __init__(self, gate_channels, groups, lay_4, pool_types=['avg', 'max']):
        super(SCA, self).__init__()
        self.lay_4 = lay_4
        if self.lay_4:
            self.lay_4_wghts = Layer_4(groups)
        else:
            self.spatial_att = SpatialGroupEnhance(groups)
            self.channel_att = ChannelGate(gate_channels)
    
            
    def forward(self, x):
        if self.lay_4:
            x_out, chan_wgt = self.lay_4_wghts(x)
            
        else:
            x_out = self.spatial_att(x)
            x_out, chan_wgt = self.channel_att(x_out)
           
        return x_out, chan_wgt


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=4, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias
    


class Layer_4(nn.Module):
    def __init__(self, groups = 64):
        super(Layer_4, self).__init__()
        self.groups   = groups
        self.compress = ChannelPool()
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): 
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn= self.compress(x)
        xn = xn.sum(dim=1, keepdim=True)
        # t = xn.view(b * self.groups, -1)
        # t = t - t.mean(dim=1, keepdim=True)
        # std = t.std(dim=1, keepdim=True) + 1e-5
        # t = t / std
        # t = t.view(b, self.groups, h, w)
        # t = t * self.weight + self.bias
        # t = t.view(b * self.groups, 1, h, w)
        xn = xn.view(b, self.groups, h, w)
        xn = xn * self.weight + self.bias
        xn = xn.view(b * self.groups, 1, h, w)
        x = x * self.sig(xn)
        x = x.view(b, c, h, w)
        
        batch_mean = torch.mean(x, dim=0)
        batch_mean = self.sig(torch.squeeze(batch_mean))
        batch_mean = batch_mean.tolist()
        
        return x, batch_mean