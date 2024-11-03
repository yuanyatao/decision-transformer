import numpy as np
import random
'''
    实现了一个简单的基于简单网格的从起点到终点的时间差分学习算法
'''

# 定义网格世界环境
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()
        
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.size - 1:
            y += 1

        self.agent_pos = (x, y)
        if self.agent_pos == self.goal:
            return self.agent_pos, 0, True  # 到达终点，奖励为 0，回合结束
        else:
            return self.agent_pos, -1, False  # 没有到终点，奖励为 -1

# 参数设置
size = 5
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
episodes = 500  # 回合数

# 初始化值函数
V = np.zeros((size, size))

# TD 学习算法
env = GridWorld(size=size)

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 随机选择一个动作
        action = random.choice(env.actions)
        
        # 与环境交互
        next_state, reward, done = env.step(action)
        
        # TD 更新
        x, y = state
        nx, ny = next_state
        td_error = reward + gamma * V[nx, ny] - V[x, y]
        V[x, y] += alpha * td_error
        
        # 更新状态
        state = next_state

# 输出结果
print("学到的状态值函数 V：")
print(V)
