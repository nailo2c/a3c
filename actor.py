# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from a3c import ActorCritic


# 確保shared model有gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None: return
        shared_param._grad = param.grad
        


def actor(rank, args, shared_model, optimizer):
    # 個別create env
    env = create_atari_env(args.env_name)
    
    # 個別訓練model
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.train()
    
    # init
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    
    while True:
        episode_length += 1
        
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        # LSTM's param
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.num_steps):
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            
            # 計算entropy
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            
            # 從multinomial抽出各個action的機率，再依據該機率取出對應的log_prob
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))
            
            # gym env step
            state, reward, done, _ = env.step(action.numpy())
            # 到termial或到達最大步數時就終止
            done = done or episode_length >= args.max_episode_length
            # 限制reward在[-1, 1]之間
            reward = max(min(reward, 1), -1)
            
            if done:
                episode_length = 0
                state = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
                
        # 實作論文中演算法的R，如果是termial state則為0
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
            
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            # advantage, 就是TD error的方向
            advantage = R - values[i]
            # 累積value loss
            value_loss += 0.5 * advantage.pow(2)
            
            # Generalized Advantage Estimation
            # 可參考原論文第4章a3c的部分，或參考以下論文的推導
            # https://arxiv.org/pdf/1506.02438.pdf 式子11~14
            delta_t = rewards[i] + args.gamma * values[i+1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            
            # 累積policy loss，但用減的還沒參透，只是先模仿
            # 初步猜測可能是log_prob與entropy都是負數的關係
            policy_loss += -(log_probs[i] * Variable(gae) + 0.01 * entropies[i])
            
        # 依據pytorch中gradient累加的特性，先將gradient歸零
        optimizer.zero_grad()
        
        # a3c的loss是將兩個loss加起來的理由應該可看這篇論文
        # https://arxiv.org/pdf/1704.06440.pdf
        (policy_loss + 0.5 * value_loss).backward()
        
        # 根據論文，實作gradient norm clipping以防止gradient explosion
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        
        # 確保share model的param有被update到
        ensure_shared_grads(model, shared_model)
        
        # update
        optimizer.step()
