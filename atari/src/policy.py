import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, state_channels, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 图片解码器：将输入图像状态转换为特征向量
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim), nn.ReLU()  # 调整线性层输出维度
        )
        # 动作网络
        self.action_dim = act_dim
        self.net = mlp([hidden_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        # 使用图片解码器编码状态
        features = self.encoder(obs)
        mean = self.net(features)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        """
        根据输入状态生成动作。
        - deterministic: 是否使用均值动作（确定性）
        - enable_grad: 是否允许梯度流动
        """
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()



class DeterministicPolicy(nn.Module):
    def __init__(self, state_channels, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 图片解码器：将输入图像状态转换为特征向量
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim), nn.ReLU()
        )
        # 动作网络
        self.action_dim = act_dim
        self.net = mlp([hidden_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        # 使用图片解码器编码状态
        features = self.encoder(obs)
        return self.net(features)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)


class DiscretePolicy(nn.Module):
    def __init__(self, state_channels, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        # 图片解码器：将输入图像状态转换为特征向量
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim), nn.ReLU()
        )
        self.action_dim = act_dim
        # 动作网络：输出每个动作的值
        self.q_net = mlp([hidden_dim, *([hidden_dim] * n_hidden), act_dim])

    def forward(self, obs):
        # 使用图片解码器编码状态
        features = self.encoder(obs)
        # 输出每个动作的 Q 值或概率分布
        return self.q_net(features)

    def act(self, obs, deterministic=True):
        """
        根据状态选择动作。
        - deterministic: 如果为 True，选择值最大的动作（贪婪策略）。
        """
        with torch.no_grad():
            q_values = self.forward(obs)  # 输出每个动作的值
            return q_values.argmax(dim=-1).cpu().numpy()  # 返回值最大的动作索引
