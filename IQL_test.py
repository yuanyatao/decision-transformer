import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 模型定义
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出为状态的值估计

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出为状态-动作对的Q值估计

    def forward(self, state, action):
        # print(state.shape)
        # print(action.shape)
        x = torch.cat([state, action], dim=-1)  # 拼接状态和动作
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # 输出为动作的分布参数

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = self.fc3(x)  # 策略网络输出动作的均值
        return action_mean

# IQL算法
class IQL:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义价值网络、Q值网络、策略网络
        self.value_network = ValueNetwork(state_dim).to(self.device)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        
        # 定义目标Q网络
        self.target_q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # 初始化为Q网络的参数
        
        # 定义优化器
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=config["value_lr"])
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config["q_lr"])
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=config["policy_lr"])
        
        # 超参数
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.tau = config["tau"]  # 用于软更新目标Q网络的参数

    def update(self, state, action, reward, next_state, done):
        # 计算TD目标值
        with torch.no_grad():
            next_action = self.policy_network(next_state)
            target_q_value = self.target_q_network(next_state, next_action)
            td_target = reward + (1 - done) * self.gamma * target_q_value

        # 更新Q网络
        q_value = self.q_network(state, action)
        q_loss = F.mse_loss(q_value, td_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 更新值函数网络
        value = self.value_network(state)
        with torch.no_grad():
            q_value_detach = self.q_network(state, action).detach()
        value_loss = F.mse_loss(value, q_value_detach)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 策略提取（Policy Extraction）
        action_pred = self.policy_network(state)
        policy_loss = -self.q_network(state, action_pred).mean()  # 最大化策略的Q值
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 软更新目标Q网络
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.policy_network(state)
        return action.detach().cpu().numpy()[0]

# 配置参数
config = {
    "value_lr": 3e-4,
    "q_lr": 3e-4,
    "policy_lr": 3e-4,
    "gamma": 0.99,
    "alpha": 0.2,
    "tau": 0.005,
}

# # 示例：定义状态维度和动作维度
# state_dim = 8
# action_dim = 2

# # 创建IQL对象
# iql_agent = IQL(state_dim, action_dim, config)

# # 示例：假设有一批经验数据
# state = torch.randn(10, state_dim).to(iql_agent.device)  # 10个状态样本
# action = torch.randn(10, action_dim).to(iql_agent.device)  # 对应的动作
# reward = torch.randn(10, 1).to(iql_agent.device)  # 奖励
# next_state = torch.randn(10, state_dim).to(iql_agent.device)  # 下一状态
# done = torch.zeros(10, 1).to(iql_agent.device)  # 是否结束标志

# # 更新IQL
# iql_agent.update(state, action, reward, next_state, done)

# # 选择动作
# action = iql_agent.select_action(state[0].cpu().numpy())
# print("Selected action:", action)

import gym
env = gym.make('Breakout-v0', render_mode="human")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
# import gym

# # 列出所有注册的环境ID
# envs = sorted(gym.envs.registry.keys())
# print(envs)
