import torch
import torch.nn as nn
from .util import mlp


# class TwinQ(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
#         self.q1 = mlp(dims, squeeze_output=True)
#         self.q2 = mlp(dims, squeeze_output=True)

#     def both(self, state, action):
#         sa = torch.cat([state, action], 1)
#         return self.q1(sa), self.q2(sa)

#     def forward(self, state, action):
#         return torch.min(*self.both(state, action))
class TwinQ(nn.Module):
    def __init__(self, state_channels, action_dim, hidden_dim=256, n_hidden=2):
        """
        - state_channels: 输入图片的通道数（例如 4 表示 4 帧叠加的图像输入）。
        - action_dim: 动作空间的维度。
        """
        super().__init__()
        # 图片解码器，用于将图像状态嵌入到特征向量中
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim), nn.ReLU()  # 将特征展平为向量
        )
        self.action_dim = action_dim
        # Q函数网络
        dims = [hidden_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        # print('hidden_dim + action_dim=',hidden_dim + action_dim)
        self.q1 = mlp(dims, squeeze_output=True)  # 第一个 Q 网络
        self.q2 = mlp(dims, squeeze_output=True)  # 第二个 Q 网络

    def encode_state(self, state):
        """
        使用图片解码器将输入图片状态转化为特征向量。
        - state: 输入的图像状态，形状为 (batch_size, channels, height, width)。
        """
        return self.encoder(state)
    
    def one_hot_encoding(self,actions):
        """
        将动作从标量转换为 one-hot 编码。
        :param actions: 动作张量，形状为 [batch_size, 1]
        :param num_actions: 动作空间的大小
        :return: one-hot 编码的动作张量，形状为 [batch_size, num_actions]
        """
        return torch.nn.functional.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()

    def both(self, state, action):
        """
        同时计算两个 Q 网络的输出。
        - state: 输入状态，形状为 (batch_size, channels, height, width)。
        - action: 输入动作，形状为 (batch_size, action_dim)。
        """
        # 编码状态
        state_features = self.encode_state(state).view(state.size(0), -1)  # 展平成 (batch_size, feature_dim)
        # 确保 action 是 2D
        if action.dim() == 1:  # 如果动作是 1D，将其扩展为 2D
            action = action.unsqueeze(-1)  # 将 (batch_size,) 转换为 (batch_size, 1)
        action_one_hot = self.one_hot_encoding(action)  # 转换为 one-hot
        # 拼接状态特征与动作
        sa = torch.cat([state_features, action_one_hot], dim=1)
        # 分别计算两个 Q 网络的输出
        return self.q1(sa), self.q2(sa)
    def forward(self, state, action):
        """
        返回两个 Q 网络的最小值，作为最终的 Q 值。
        """
        return torch.min(*self.both(state, action))


### 修改编码器，用于训练Atari数据集
class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),  # 类似DT的卷积层
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim), nn.ReLU()  # 将图片特征压缩为向量
        )
        self.v = mlp([hidden_dim, *([hidden_dim] * n_hidden), 1], squeeze_output=True)

    def forward(self, state):
        features = self.encoder(state)
        return self.v(features)