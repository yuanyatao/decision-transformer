import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)  # 随机种子
parser.add_argument('--context_length', type=int, default=30)  # 上下文长度，用于序列建模
parser.add_argument('--epochs', type=int, default=1)  # 训练的轮数
parser.add_argument('--model_type', type=str, default='reward_conditioned')  # 模型类型
parser.add_argument('--num_steps', type=int, default=500000)  # 训练中使用的步数
parser.add_argument('--num_buffers', type=int, default=50)  # 数据缓冲区的数量
parser.add_argument('--game', type=str, default='Breakout')  # 使用的游戏环境
parser.add_argument('--batch_size', type=int, default=4)  # 训练的批次大小 128
# 每个缓冲区中抽样的轨迹数
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')  # 数据目录的前缀
parser.add_argument('--num_samples', type=int, default=1000)  # 随机生成的样本数量
args = parser.parse_args()

set_seed(args.seed)
# 定义一个用于存储状态、动作和回报数据的数据集类
class RandomDataset(Dataset):
    def __init__(self, num_samples, context_length):
        self.data = torch.rand(num_samples, 4 * 84 * 84)  # 随机生成状态
        self.actions = torch.randint(0, 5, (num_samples, 1))  # 随机生成动作，假设有5个动作
        self.rtgs = torch.rand(num_samples, 1)  # 随机生成回报-to-go
        self.timesteps = torch.arange(num_samples).unsqueeze(1)  # 时间步
        self.done_idxs = (torch.arange(num_samples) // context_length) * context_length  # 随机生成结束索引
        self.block_size = context_length * 3  # 数据块大小

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        states = self.data[idx:idx + self.block_size].reshape(self.block_size, -1)
        actions = self.actions[idx:idx + self.block_size]
        rtgs = self.rtgs[idx:idx + self.block_size]
        timesteps = self.timesteps[idx:idx + self.block_size]

        return states, actions, rtgs, timesteps
# 定义一个用于存储状态、动作和回报数据的数据集类
class StateActionReturnDataset(Dataset):
    # 初始化函数
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size  # 数据块的大小
        self.vocab_size = max(actions) + 1  # 动作的词汇大小
        self.data = data  # 状态数据
        self.actions = actions  # 动作数据
        self.done_idxs = done_idxs  # 每个轨迹结束的位置索引
        self.rtgs = rtgs  # 回报-to-go 的数据
        self.timesteps = timesteps  # 时间步数据
    
    # 获取数据集的长度
    def __len__(self):
        return len(self.data) - self.block_size

    # 获取指定索引的数据块
    def __getitem__(self, idx):
        block_size = self.block_size // 3  # 将数据块大小按比例划分
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # 找到第一个大于当前索引的 done_idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        # 提取状态数据，并转换为张量 (block_size, 4*84*84)，进行归一化处理
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        states = states / 255.
        # 提取动作数据，并转换为张量 (block_size, 1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        # 提取回报-to-go 并转换为张量
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        # 提取时间步数据
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

# 使用 create_dataset 函数创建数据集，返回状态、动作、回报、结束索引、回报-to-go 和时间步数据
# obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# 设置日志记录的格式和级别
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# # 创建训练数据集实例
# # train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
# # 创建随机数据集实例
# train_dataset = RandomDataset(args.num_samples, args.context_length)
# # 配置 GPT 模型的参数
# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                   n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
# # 初始化 GPT 模型
# model = GPT(mconf)

# # 配置训练器的参数
# tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=6e-4,
#                       lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
#                       num_workers=0, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
# # 初始化训练器实例，并开始训练
# trainer = Trainer(model, train_dataset, None, tconf)

# # 开始模型的训练过程
# trainer.train()
# 创建随机数据集实例
train_dataset = RandomDataset(args.num_samples, args.context_length)

# 配置 GPT 模型的参数
mconf = GPTConfig(train_dataset.actions.max() + 1, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=train_dataset.timesteps.max())
# 初始化 GPT 模型
model = GPT(mconf)

# 配置训练器的参数
tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512 * 20, final_tokens=2 * len(train_dataset) * args.context_length * 3,
                      num_workers=0, seed=args.seed, model_type=args.model_type)

# 初始化训练器实例，并开始训练
trainer = Trainer(model, train_dataset, None, tconf)

# 开始模型的训练过程
trainer.train()