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

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    """
    创建强化学习数据集，从固定的回放缓冲区中加载数据。

    参数：
    - num_buffers: 使用的缓冲区数量
    - num_steps: 需要加载的总步数
    - game: 训练使用的游戏名称
    - data_dir_prefix: 数据目录的前缀
    - trajectories_per_buffer: 每个缓冲区中要加载的轨迹数量

    返回：
    - obss: 状态观测序列
    - actions: 动作序列
    - returns: 每个轨迹的累计回报
    - done_idxs: 每个轨迹结束时的索引
    - rtg: 每个时间步的回报-to-go（未来回报的累计值）
    - timesteps: 每个动作对应的时间步
    """
    # 初始化存储状态观测、动作、回报、轨迹终点索引和逐步回报的列表
    obss = []
    actions = []
    returns = [0]  # 初始化为0，以便累加
    done_idxs = []
    stepwise_returns = []

    # 创建一个数组来跟踪每个缓冲区已加载的转移数
    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0  # 记录已加载的轨迹数量

    # 当加载的观测数少于要求的步数时，继续加载
    while len(obss) < num_steps:
        # 从可用的缓冲区中随机选择一个缓冲区编号进行数据加载
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]  # 获取当前缓冲区已加载的转移数
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))

        # 创建一个固定回放缓冲区实例，从中加载数据
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',  # 数据目录
            replay_suffix=buffer_num,  # 缓冲区编号
            observation_shape=(84, 84),  # 状态观测的形状
            stack_size=4,  # 每个状态由4帧堆叠而成
            update_horizon=1,
            gamma=0.99,  # 折扣因子
            observation_dtype=np.uint8,  # 状态观测的类型
            batch_size=32,  # 批次大小
            replay_capacity=100000  # 缓冲区的最大容量
        )
        
        # 如果缓冲区中有已加载的数据，则继续加载
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)  # 当前已加载的总转移数
            trajectories_to_load = trajectories_per_buffer  # 剩余需要加载的轨迹数

            # 开始从缓冲区加载数据，直到加载足够的轨迹
            while not done:
                # 从回放缓冲区中抽取一个转移样本
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                # 转置状态，使其形状从 (1, 84, 84, 4) 转换为 (4, 84, 84)，并取出第一个样本
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]  # 将状态添加到观测列表中
                actions += [ac[0]]  # 添加动作到动作列表
                stepwise_returns += [ret[0]]  # 添加单步回报到逐步回报列表

                # 如果当前状态是终止状态（即该轨迹结束）
                if terminal[0]:
                    done_idxs += [len(obss)]  # 记录终止状态的索引
                    returns += [0]  # 为下一个轨迹初始化回报
                    if trajectories_to_load == 0:  # 如果所有轨迹已加载完毕
                        done = True  # 结束当前缓冲区的加载
                    else:
                        trajectories_to_load -= 1  # 减少剩余需要加载的轨迹数

                returns[-1] += ret[0]  # 更新当前轨迹的累计回报
                i += 1  # 增加当前缓冲区的已加载转移数

                # 如果超过了缓冲区容量，则回退到上次加载的位置，并结束
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0  # 重置当前回报
                    i = transitions_per_buffer[buffer_num]  # 回到之前的位置
                    done = True

            # 增加已加载的轨迹数量
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            # 更新缓冲区已加载的转移数
            transitions_per_buffer[buffer_num] = i

        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    # 将动作、回报、逐步回报和轨迹终点索引转换为 NumPy 数组
    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- 创建回报-to-go 数据集（即未来累计回报）
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)  # 初始化 rtg 数组，大小与逐步回报相同
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]  # 当前轨迹的逐步回报
        # 从轨迹终点开始，倒序计算每一步的 rtg
        for j in range(i-1, start_index-1, -1):  # 从 i-1 开始
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)  # 计算当前步的 rtg
        start_index = i  # 更新下一个轨迹的起点
    print('max rtg is %d' % max(rtg))  # 打印最大回报-to-go

    # -- 创建时间步数据集
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)  # 初始化时间步数组，长度为动作数量加1
    for i in done_idxs:
        i = int(i)
        # 为当前轨迹的每一步分配时间步，逐步递增
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1  # 更新下一个轨迹的起点
    print('max timestep is %d' % max(timesteps))  # 打印最大时间步

    # 返回所有生成的数据
    return obss, actions, returns, done_idxs, rtg, timesteps