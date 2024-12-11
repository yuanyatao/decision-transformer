import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal  # 导入多元正态分布类

from .util import mlp  # 从自定义模块中导入多层感知机（MLP）函数

# 定义对数标准差的上下限，用于约束标准差的取值范围
LOG_STD_MIN = -5.0  # 对数标准差的最小值
LOG_STD_MAX = 2.0   # 对数标准差的最大值

# 高斯策略类，基于多元正态分布
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        """
        初始化高斯策略网络。

        参数：
        - obs_dim: 观察空间维度
        - act_dim: 动作空间维度
        - hidden_dim: 隐藏层神经元的数量，默认为256
        - n_hidden: 隐藏层的数量，默认为2
        """
        super().__init__()
        # 创建MLP网络，输入为观察空间维度，输出为动作空间维度
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        # 定义一个可训练参数log_std（对数标准差），初始值为全零
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        """
        前向传播，根据输入的观察值生成动作分布。

        参数：
        - obs: 输入的观察值（tensor）

        返回：
        - 多元正态分布（MultivariateNormal）对象
        """
        mean = self.net(obs)  # 通过MLP网络计算动作的均值
        # 计算标准差，并将对数标准差限制在预设范围内
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        # 构造对角矩阵，表示多元正态分布的协方差矩阵
        scale_tril = torch.diag(std)
        # 返回多元正态分布对象
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        """
        根据当前策略生成动作。

        参数：
        - obs: 输入的观察值（tensor）
        - deterministic: 是否以确定性方式选择动作（默认为False，即随机采样）
        - enable_grad: 是否启用梯度计算（默认为False）

        返回：
        - 动作（tensor）
        """
        with torch.set_grad_enabled(enable_grad):  # 根据enable_grad参数决定是否计算梯度
            dist = self(obs)  # 根据观察值生成多元正态分布
            return dist.mean if deterministic else dist.sample()  # 返回均值或随机样本


# 确定性策略类
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        """
        初始化确定性策略网络。

        参数：
        - obs_dim: 观察空间维度
        - act_dim: 动作空间维度
        - hidden_dim: 隐藏层神经元的数量，默认为256
        - n_hidden: 隐藏层的数量，默认为2
        """
        super().__init__()
        # 创建MLP网络，最后一层激活函数为Tanh，用于限制动作范围
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        """
        前向传播，根据输入的观察值生成确定性动作。

        参数：
        - obs: 输入的观察值（tensor）

        返回：
        - 确定性动作（tensor）
        """
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        """
        根据当前策略生成动作。

        参数：
        - obs: 输入的观察值（tensor）
        - deterministic: 是否以确定性方式选择动作（默认为False）
        - enable_grad: 是否启用梯度计算（默认为False）

        返回：
        - 动作（tensor）
        """
        with torch.set_grad_enabled(enable_grad):  # 根据enable_grad参数决定是否计算梯度
            return self(obs)  # 返回通过网络计算的确定性动作
