import csv
import logging
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
import argparse

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)  # 随机种子
parser.add_argument('--context_length', type=int, default=30)  # 上下文长度，用于序列建模
parser.add_argument('--epochs', type=int, default=1)  # 训练的轮数
parser.add_argument('--model_type', type=str, default='reward_conditioned')  # 模型类型
parser.add_argument('--num_samples', type=int, default=1000)  # 随机生成的样本数量
parser.add_argument('--batch_size', type=int, default=4)  # 训练的批次大小
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
        # 注意这里确保生成的 timesteps 数量匹配 block_size
        states = self.data[idx:idx + self.block_size].reshape(self.block_size, -1)
        actions = self.actions[idx:idx + self.block_size]
        rtgs = self.rtgs[idx:idx + self.block_size]
        timesteps = self.timesteps[idx:idx + self.block_size]

        return states, actions, rtgs, timesteps
# 设置日志记录的格式和级别
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

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
