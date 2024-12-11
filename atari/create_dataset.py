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
from fixed_replay_buffer import FixedReplayBuffer

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, mode="DT"):
    """
    创建强化学习数据集，从固定的回放缓冲区中加载数据。

    参数：
    - num_buffers: 使用的缓冲区数量
    - num_steps: 需要加载的总步数
    - game: 训练使用的游戏名称
    - data_dir_prefix: 数据目录的前缀
    - trajectories_per_buffer: 每个缓冲区中要加载的轨迹数量
    - mode: 数据模式，支持 "DT" 和 "IQL"，控制返回数据的内容和格式

    返回：
    根据模式返回以下数据：
    - DT: obss, actions, returns, done_idxs, rtg, timesteps
    - IQL: obss, actions, rewards, terminals
    """
    obss = []
    actions = []
    returns = [0]  # 初始化为0，以便累加
    done_idxs = []
    stepwise_returns = []
    next_obss = []  # 添加 next_obss，用于存储下一状态
    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0

    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]

        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000
        )
        
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer

            while not done:
                # 从回放缓冲区中抽取一个转移样本
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                # 转置状态，使其形状从 (1, 84, 84, 4) 转换为 (4, 84, 84)，并取出第一个样本
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                next_states = next_states.transpose((0, 3, 1, 2))[0]
                obss += [states]  # 将状态添加到观测列表中
                next_obss += [next_states]  # 将状态添加到观测列表中
                actions += [ac[0]]  # 添加动作到动作列表
                stepwise_returns += [ret[0]]  # 添加单步回报到逐步回报列表

                # 如果当前状态是终止状态（即该轨迹结束）
                if terminal[0]:
                    done_idxs += [len(obss)]  # 记录终止状态的索引
                    returns += [0]  # 为下一个轨迹初始化回报
                    if trajectories_to_load == 0:  # 如果所有轨迹已加载完毕
                        done = True  # 结束当前缓冲区的加载
                    else:
                        trajectories_to_load -= 1

                returns[-1] += ret[0]  # 更新当前轨迹的累计回报
                i += 1  # 增加当前缓冲区的已加载转移数

                # 如果超过了缓冲区容量，则回退到上次加载的位置，并结束
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    next_obss = next_obss[:curr_num_transitions]  # 修剪 next_obss
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0  # 重置当前回报
                    i = transitions_per_buffer[buffer_num]  # 回到之前的位置
                    done = True

            # 增加已加载的轨迹数量
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            # 更新缓冲区已加载的转移数
            transitions_per_buffer[buffer_num] = i

        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' %
              (i, len(obss), num_trajectories))

    # 将动作、回报、逐步回报和轨迹终点索引转换为 NumPy 数组
    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)
    print('actions.shape:',actions.shape,'actions[:5]:',actions[:5])
    print('returns.shape:',returns.shape)
    start_index = 0
    # -- 创建回报-to-go 数据集（即未来累计回报）

    rtg = np.zeros_like(stepwise_returns)  # 初始化 rtg 数组，大小与逐步回报相同
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):
            rtg[j] = sum(curr_traj_returns[j - start_index:i - start_index])
        start_index = i


    if mode == "DT":
        timesteps = np.zeros(len(actions) + 1, dtype=int)
        start_index = 0
        for i in done_idxs:
            i = int(i)
            timesteps[start_index:i + 1] = np.arange(i + 1 - start_index)
            start_index = i + 1
        print('max timestep is %d' % max(timesteps))
        return obss, actions, returns, done_idxs, rtg, timesteps

    elif mode == "IQL":
        # 添加适配 IQL 的终止标志和单步奖励
        terminals = np.zeros_like(actions, dtype=np.float32)
        terminals[done_idxs - 1] = 1.0  # 将终止状态标志置为 1
        rewards = stepwise_returns  # IQL 直接使用单步回报作为奖励
        print('IQL加载完毕')
        return obss, next_obss,actions, rewards, terminals

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose either 'DT' or 'IQL'.")