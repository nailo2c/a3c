# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    # 除以norm 2，再乘以std來控制拉伸的長度
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out



# Xavier initialization
def weights_init(m):
    classname = m.__class__.__name__
    # 對捲積層做initialization
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    
    # 對FC做initialization
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)



class ActorCritic(torch.nn.Module):
    
    def __init__(self, num_inputs, action_space):
        
        super(ActorCritic, self).__init__()
        
        # 捲積層
        # 42x42, 42+2=44, 捲成 21x21
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        # 21x21, 21+2=23, 捲成 11x11
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # 11x11, 11+2=13, 捲成 6x6
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # 6x6, 6+2=8, 捲成 3x3 
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        # LSTM
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        
        # actor-critic
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        
        # 權重初始化 & normalized
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        
        # bias為0
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        # training mode
        self.train()
        
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        
        x = x.view(-1, 32 * 3 * 3) # 展開乘1x288的向量
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)



# 參考 https://github.com/openai/universe-starter-agent
class SharedAdam(optim.Adam):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        
        super(SharedAdam, self).__init__(params, lr, betas, eps)
        
        # init to 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    # share adam's param
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    # update weight
    def step(self):
        loss = None
        
        for group in self.param_groups:
            for p in group['params']:
                # 檢查p是否有gradient，若沒有則進行下一個迴圈
                if p.grad is None: continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # 提取adam的參數
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # update first moment estimate & second moment estimate
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1
                
                # inplce mode of addcdiv
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return loss
