"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
import torch.nn.functional as F

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        dt_loss_history = []
        policy_loss_history = []
        total_loss_history = []
        # 记录每个 batch 的损失
        all_batch_dt_losses = []
        all_batch_policy_losses = []
        all_batch_total_losses = []
        all_eval_returns = []  # 用于记录每个 epoch 的评估得分
        # 获取模型和配置文件
        model, config = self.model, self.config
        # 如果模型有多个模块（比如使用了DataParallel），则获取模块中的模型，否则直接使用模型
        raw_model = model.module if hasattr(self.model, "module") else model
        # 根据配置文件设置优化器
        optimizer = raw_model.configure_optimizers(config)

        # 定义一个运行 epoch（训练轮次）的方法，split表示数据集的类型，epoch_num是当前的轮次数
        def run_epoch(split, epoch_num=0, dt_loss_history=None, policy_loss_history=None, total_loss_history=None, batch_loss_history=None):
            # 记录每个 batch 的损失
            batch_dt_losses = []      # 记录每个 batch 的 DT 损失
            batch_policy_losses = []  # 记录每个 batch 的策略损失
            batch_total_losses = []   # 记录每个 batch 的总损失
            
            # 判断是训练模式还是测试模式
            is_train = split == 'train'
            # 设置模型的模式，train(True)为训练模式，train(False)为评估模式
            model.train(is_train)
            # 选择训练集或测试集
            data = self.train_dataset if is_train else self.test_dataset
            # 使用DataLoader加载数据集，设置数据加载时是否打乱顺序以及一些并行处理的参数
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            # 保存每个batch的损失值
            losses = []
            # 如果是训练模式，使用tqdm显示进度条；否则只枚举数据
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # 遍历数据加载器，it是当前迭代次数，(x, y, r, t)是加载的数据
            for it, (x, y, r, t) in pbar:
                # 将数据移动到指定的设备（如GPU）
                x = x.to(self.device) #states: 状态输入 (batch, block_size, 4*84*84) torch.Size([128, 30, 28224])
                y = y.to(self.device) # actions: 动作输入 (batch, block_size, 1) torch.Size([128, 30, 1])
                r = r.to(self.device) # rtgs: 回报-to-go (batch, block_size, 1) torch.Size([128, 30, 1])
                t = t.to(self.device) # timesteps: 时间步 (batch, 1, 1) 

                # 通过模型进行前向传播
                with torch.set_grad_enabled(is_train):  # 只有在训练模式下才启用梯度计算
                    # logits, dt_loss = model(x, y, y, r, t)  # 使用loss
                    logits, dt_loss, q_values = model(x, y, y, r, t)
                    # print('x.shape',x.shape)
                    # print('y.shape',y.shape)
                    # print('r.shape',r.shape)
                    # print('t.shape',t.shape)
                    # print('logits.shape',logits.shape) # torch.Size([128, 30, 4])
                    # print('dt_loss.shape',dt_loss.shape) # torch.Size([])
                    # print('q_values.shape',q_values.shape)
                    dt_loss = dt_loss.mean()  # 确保损失是标量
                    batch_dt_losses.append(dt_loss.item())
                    # 以下是新增
                    # 计算 Q 值

                    q_values = q_values.squeeze(-1)  # 形状变为 [batch_size, sequence_length]
                    baseline = q_values.mean(dim=1, keepdim=True)
                    advantages = q_values - baseline  # 计算优势
                    weights = torch.exp(advantages)
                    # 利用优势强化 logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    print("log_probs shape:", log_probs.shape)  # (batch_size, sequence_length, vocab_size)
                    print("y shape:", y.shape)                  # (batch_size, sequence_length)
                    # 将 y 调整为 [batch_size, sequence_length]
                    # y = y.squeeze(-1)
                    chosen_log_probs = log_probs.gather(2, y).squeeze(-1)  # 输出维度 [batch_size, sequence_length]

                    policy_loss = -(weights * chosen_log_probs).mean(dim=-1).mean()  # 优先对 sequence_length 平均

                    # policy_loss = -(advantages * chosen_log_probs).mean()  # 策略损失
                    batch_policy_losses.append(policy_loss.item())
                    # 使用优势加权计算最终损失 加负数的原因是 要最大化优势 就要最小化损失 
                    total_loss = policy_loss + dt_loss  # 加上行为克隆损失
                    batch_total_losses.append(total_loss.item())



                if is_train:
                    # 如果是训练模式，则执行反向传播和参数更新
                    model.zero_grad()  # 清除前一轮的梯度
                    total_loss.backward()  # 反向传播计算梯度
                    # 防止梯度爆炸，进行梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()  # 更新参数

                    # 根据训练进度调整学习率
                    if config.lr_decay:
                        # 计算当前处理的tokens数量（不包括-100的标签）
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            # 在预热阶段进行线性学习率上升
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # 在预热之后使用余弦衰减策略调整学习率
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        # 根据学习率的倍数调整学习率
                        lr = config.learning_rate * lr_mult
                        # 为优化器中的每个参数组设置新的学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        # 如果不使用学习率衰减，则直接使用初始学习率
                        lr = config.learning_rate

                    # 更新进度条，显示当前迭代的损失值和学习率
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {total_loss.item():.5f}. lr {lr:e}")

            # 如果是测试模式，计算平均损失并记录日志
            if not is_train:
                test_loss = float(np.mean(batch_total_losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            # 记录 batch 级别的损失
            if dt_loss_history is not None:
                dt_loss_history.append(np.mean(batch_dt_losses))
            if policy_loss_history is not None:
                policy_loss_history.append(np.mean(batch_policy_losses))
            if total_loss_history is not None:
                total_loss_history.append(np.mean(batch_total_losses))

            return batch_dt_losses, batch_policy_losses, batch_total_losses  # 返回每个 batch 的损失
                # best_return用于记录当前最好的返回值，初始值为负无穷
        best_return = -float('inf')

        # 初始化token计数器，用于学习率衰减
        self.tokens = 0

        # 遍历所有的训练轮次
        for epoch in range(config.max_epochs):
            # 运行训练模式的epoch
            batch_dt_losses, batch_policy_losses, batch_total_losses = run_epoch('train', epoch_num=epoch, 
                                                                                   dt_loss_history=dt_loss_history,
                                                                                   policy_loss_history=policy_loss_history,
                                                                                   total_loss_history=total_loss_history)        # 将每个 epoch 的 batch 损失添加到总列表中
            all_batch_dt_losses.extend(batch_dt_losses)
            all_batch_policy_losses.extend(batch_policy_losses)
            all_batch_total_losses.extend(batch_total_losses)

            # 以下是根据模型类型进行的评估流程
            if self.config.model_type == 'naive':
                # 如果模型类型是'naive'，调用self.get_returns(0)获取返回值
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                # 如果模型类型是'reward_conditioned'，根据不同的游戏环境获取对应的返回值
                if self.config.game == 'Breakout':
                    eval_return = self.get_returns(90)
                elif self.config.game == 'Seaquest':
                    eval_return = self.get_returns(1150)
                elif self.config.game == 'Qbert':
                    eval_return = self.get_returns(14000)
                elif self.config.game == 'Pong':
                    eval_return = self.get_returns(20)
                else:
                    # 如果遇到未实现的游戏环境，抛出错误
                    raise NotImplementedError()
                

            else:
                # 如果遇到未实现的模型类型，抛出错误
                raise NotImplementedError()
            all_eval_returns.append(eval_return)
            best_return = max(eval_return,best_return)
        self.save_train_result(dt_loss_history, policy_loss_history, total_loss_history, all_eval_returns)
        print(f'best_return in total {config.max_epochs} epochs:{best_return}')
        
    def save_train_result(self, dt_losses, policy_losses, total_losses, eval_returns):
        # Step 1: 创建 train_result 目录和 exp_n 目录
        base_dir = "train_result"
        os.makedirs(base_dir, exist_ok=True)

        # 查找下一个可用的 exp_n 目录
        exp_num = 1
        exp_dir = os.path.join(base_dir, f"exp_{exp_num}")
        while os.path.exists(exp_dir):
            exp_num += 1
            exp_dir = os.path.join(base_dir, f"exp_{exp_num}")
        os.makedirs(exp_dir)

        # Step 2: 绘制并保存损失对比图
        plt.figure(figsize=(12, 6))
        plt.plot(dt_losses, label="DT Loss", color="blue")
        plt.plot(policy_losses, label="Policy Loss", color="orange")
        plt.plot(total_losses, label="Total Loss", color="green")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        plt.title("Loss per Batch during Training")
        plt.legend()
        
        # 保存为 SVG 格式
        plt.savefig(os.path.join(exp_dir, "losses_comparison.svg"), format='svg')
        plt.close()

        # Step 3: 保存每个 epoch 的 eval_return 到 score.txt
        score_file_path = os.path.join(exp_dir, "score.txt")
        with open(score_file_path, 'w') as f:
            for epoch, return_value in enumerate(eval_returns, start=1):
                f.write(f"epoch_{epoch} {return_value}\n")
            
            best_score = max(eval_returns)
            f.write(f"best_score {best_score}\n")

        print(f'整个训练结果保存成功，保存在:{exp_dir}')
        
    def get_returns(self, ret):
        # 将模型设置为评估模式，不启用梯度计算（train(False)），避免训练时的dropout或其他影响
        self.model.train(False)    
        # 初始化游戏环境的参数，包含游戏名称（转为小写）和随机种子
        args = Args(self.config.game.lower(), self.config.seed)
        # 创建一个环境实例，env表示游戏环境
        env = Env(args)
        # 将环境设置为评估模式
        env.eval()

        # 初始化两个列表，用于存储每次迭代的奖励（T_rewards）和Q值（T_Qs）
        T_rewards, T_Qs = [], []
        
        # done 表示是否完成一局游戏，初始设为True，以便开始新的游戏
        done = True
        
        # 执行10次评估迭代，每次迭代代表一次独立的游戏过程
        for i in range(10):
            # 重置环境，获得初始状态
            state = env.reset()
            # 将状态转换为浮点型，并移动到指定的设备（如GPU），并调整维度（增加批次维度和时间步维度）
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            
            # 设置初始的期望回报值，rtgs 表示期望的累计回报（Return-To-Go）
            rtgs = [ret]
            
            # 使用模型进行采样，获取初始的动作。输入参数包括：
            # - 模型
            # - 初始状态（all_states）
            # - 采样长度（1）
            # - 采样温度（temperature=1.0）
            # - 是否进行采样（sample=True）
            # - 动作（actions=None，因为是第一次采样）
            # - rtgs：期望的回报值，经过格式化后转为tensor
            # - timesteps：当前的时间步（这里是0，因为是初始步）
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            # 初始化计数器 j，用于记录当前的时间步
            j = 0
            # all_states 用于保存所有的状态（包括初始状态），动作列表actions用于保存每次采样的动作
            all_states = state
            actions = []
            
            # 开始执行游戏，直到游戏结束（done为True）
            while True:
                if done:
                    # 如果游戏已结束，重置环境，reward_sum重置为0，并将done设置为False开始新一轮
                    state, reward_sum, done = env.reset(), 0, False
                
                # 将采样得到的动作转为numpy数组，并取最后一个动作
                action = sampled_action.cpu().numpy()[0, -1]
                
                # 记录当前采样的动作
                actions += [sampled_action]
                
                # 使用当前动作在环境中执行一步，得到新的状态、奖励和游戏是否结束的标志
                state, reward, done = env.step(action)
                # 更新累计的奖励
                reward_sum += reward
                # 时间步增加
                j += 1

                # 如果游戏结束，则记录累计奖励并跳出循环
                if done:
                    T_rewards.append(reward_sum)
                    break

                # 更新状态，将新获得的状态添加到状态序列中
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                all_states = torch.cat([all_states, state], dim=0)

                # 更新rtgs，新的rtg是之前的rtg减去这一步的奖励
                rtgs += [rtgs[-1] - reward]

                # 继续使用模型进行采样，更新动作
                # - 输入包含所有的状态（all_states）
                # - 期望回报（rtgs）
                # - 动作序列（actions）
                # - 当前时间步（使用min(j, self.config.max_timestep)来限制时间步的最大值）
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
        
        # 关闭游戏环境，释放资源
        env.close()
        
        # 计算10次游戏的平均奖励（作为评估结果）
        eval_return = sum(T_rewards) / 10.
        
        # 打印目标回报（ret）和评估得到的回报（eval_return）
        print("target return: %d, eval return: %d" % (ret, eval_return))
        
        # 恢复模型到训练模式
        self.model.train(True)
        
        # 返回评估得到的平均回报值
        return eval_return


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
