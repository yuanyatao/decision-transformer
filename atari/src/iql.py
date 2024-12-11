import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average

# 定义指数优势的最大值，用于限制优势加权的影响，防止数值过大
EXP_ADV_MAX = 100.

# 不对称 L2 损失函数，用于计算价值函数的损失
def asymmetric_l2_loss(u, tau):
    """
    u: 差值（例如优势）向量
    tau: 不对称损失的权重参数，用于对正负误差施加不同权重
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        """
        初始化 IQL 模型
        - qf: Q 函数网络，用于估计状态-动作对的价值
        - vf: 价值函数网络，用于估计状态的期望价值
        - policy: 策略网络，用于输出给定状态的动作分布
        - optimizer_factory: 优化器生成函数，用于创建优化器
        - max_steps: 训练的最大步数，用于学习率调度
        - tau: 不对称 L2 损失的权重参数
        - beta: 优势加权的参数，控制策略损失中的优势影响
        - discount: 折扣因子，控制未来回报的折扣
        - alpha: 目标 Q 网络的平滑更新率
        """
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)  # Q 函数
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)  # 目标 Q 网络，防止训练时更新
        self.vf = vf.to(DEFAULT_DEVICE)  # 价值函数
        self.policy = policy.to(DEFAULT_DEVICE)  # 策略网络
        self.v_optimizer = optimizer_factory(self.vf.parameters())  # 价值函数的优化器
        self.q_optimizer = optimizer_factory(self.qf.parameters())  # Q 函数的优化器
        self.policy_optimizer = optimizer_factory(self.policy.parameters())  # 策略网络的优化器
        # 使用余弦退火调度策略来调整策略网络的学习率
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        
    def one_hot_encoding(self,actions, num_actions):
        """
        将离散动作从标量转换为 one-hot 编码。
        """
        return torch.nn.functional.one_hot(actions, num_classes=num_actions).float()
    
    def update(self, observations, next_observations,actions ,rewards, terminals):
        """
        执行一次 IQL 更新，包括价值函数、Q 函数、目标 Q 网络和平滑更新以及策略更新。
        - observations: 当前状态
        - actions: 执行动作
        - next_observations: 下一状态
        - rewards: 即时奖励
        - terminals: 终止标志（用于处理终止状态）
        """
        # print('actions.shape in update():',actions.shape)
        # print('observations.shape in update():',observations.shape)
        # print('next_observations.shape in update():',next_observations.shape)
        # print('rewards.shape in update():',rewards.shape)
        # print('terminals.shape in update():',terminals.shape)
        
        with torch.no_grad():  # 在目标计算中不进行梯度计算
            target_q = self.q_target(observations, actions)  # 使用目标 Q 网络计算当前状态-动作的目标 Q 值
            next_v = self.vf(next_observations)  # 计算下一个状态的价值估计

        # 价值函数更新
        v = self.vf(observations)  # 当前状态的价值估计
        adv = target_q - v  # 计算当前状态下的优势 A(s, a) = Q(s, a) - V(s)
        v_loss = asymmetric_l2_loss(adv, self.tau)  # 使用不对称 L2 损失计算价值函数损失
        self.v_optimizer.zero_grad(set_to_none=True)  # 清空价值函数梯度
        v_loss.backward()  # 反向传播计算梯度
        self.v_optimizer.step()  # 更新价值函数网络的参数

        # Q 函数更新
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        # 计算 TD 目标值，折扣并加上下一状态的价值
        qs = self.qf.both(observations, actions)  # 获取 Q 函数网络的输出（可能有多个头，防止单一估计的不稳定）
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)  # 均方误差 (MSE) 损失，多个 Q 值平均
        self.q_optimizer.zero_grad(set_to_none=True)  # 清空 Q 函数梯度
        q_loss.backward()  # 反向传播计算 Q 函数的梯度
        self.q_optimizer.step()  # 更新 Q 函数网络的参数

        # 目标 Q 网络的平滑更新，类似 Polyak 平均
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # 策略更新
        # 将动作转换为 one-hot 编码
        actions_one_hot = self.one_hot_encoding(actions, self.policy.action_dim)
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)  # 计算优势的指数权重，限制最大值
        policy_out = self.policy(observations)  # 策略网络输出
        if isinstance(policy_out, torch.distributions.Distribution):
            # 如果策略输出是概率分布，使用 log_prob 计算行为克隆损失
            bc_losses = -policy_out.log_prob(actions_one_hot)
        elif torch.is_tensor(policy_out):
            # 如果策略输出是张量，直接计算与执行动作的均方误差
            assert policy_out.shape == actions_one_hot.shape
            bc_losses = torch.sum((policy_out - actions_one_hot)**2, dim=1)
        else:
            raise NotImplementedError
        # 计算策略损失：行为克隆损失加权优势，鼓励选择高优势动作
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)  # 清空策略网络梯度
        policy_loss.backward()  # 反向传播计算策略损失的梯度
        self.policy_optimizer.step()  # 更新策略网络的参数
        self.policy_lr_schedule.step()  # 更新策略的学习率
        
        # 计算总损失并返回
        total_loss = v_loss.item() + q_loss.item() + policy_loss.item()
        losses = {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item()
        }
        return total_loss, losses
