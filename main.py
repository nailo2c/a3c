import argparse
import os
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from envs import create_atari_env
from a3c import ActorCritic, SharedAdam
from actor import actor
from monitor import monitor


parser = argparse.ArgumentParser(description='A3C')

# 參數部分
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')


# 演算法部分
parser.add_argument('--num-processes', type=int, default=4, metavar='NP',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='maximun length of an episode (default: 100000)')
parser.add_argument('--env-name', default='Pong-v0', metavar='ENV',
                    help='environment to train on (default: Pong-v0)')



if __name__ == '__main__':
    # 控制 Thread 的數量
    os.environ['OMP_NUM_THREADS'] = '1'

    # 獲取參數
    args = parser.parse_args()

    # 創建環境
    env = create_atari_env(args.env_name)

    # Critic
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)

    # 開啟share_memory mode
    shared_model.share_memory()

    # optimizer, adam with shared statistics
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    # multiprocesses, Hogwild! style update
    # 參考 https://github.com/pytorch/examples/tree/master/mnist_hogwild

    processes = []

    # monitor, 用來觀察目前model的訓練情況
    p = mp.Process(target=monitor, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    # actor, 平行創造各個環境，各別訓練agents
    for rank in range(0, args.num_processes):
        p = mp.Process(target=actor, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)

    # join, 確保update不衝突
    for p in processes:
        p.join()
