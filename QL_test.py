import numpy as np
import gym

class QLearningAgent:
    def __init__(self, state_space, action_space, config):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space, action_space))  # 初始化Q值表，所有Q值初始为0
        self.alpha = config["alpha"]  # 学习率
        self.gamma = config["gamma"]  # 折扣因子
        self.epsilon = config["epsilon"]  # 探索率
        self.epsilon_decay = config["epsilon_decay"]  # 探索率衰减
        self.epsilon_min = config["epsilon_min"]  # 最小探索率

    def choose_action(self, state):
        # 通过 ε-greedy 策略选择动作
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # 随机选择动作（探索）
        else:
            return np.argmax(self.q_table[state])  # 选择Q值最大的动作（利用）

    def update_q_table(self, state, action, reward, next_state, done):
        # 计算TD目标值
        td_target = reward + (1 - done) * self.gamma * np.max(self.q_table[next_state])
        # Q值更新公式
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * td_target

        # 如果回合结束，减少探索率
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 配置参数
config = {
    "alpha": 0.1,  # 学习率
    "gamma": 0.99,  # 折扣因子
    "epsilon": 1.0,  # 初始探索率
    "epsilon_decay": 0.995,  # 探索率衰减系数
    "epsilon_min": 0.01  # 最小探索率
}

# 创建环境（例如OpenAI Gym中的FrozenLake环境）
env = gym.make("FrozenLake-v1", is_slippery=False)

# 创建Q-Learning智能体
state_space = env.observation_space.n  # 状态空间大小
action_space = env.action_space.n  # 动作空间大小
agent = QLearningAgent(state_space, action_space, config)

# 训练Q-Learning智能体
num_episodes = 500  # 训练回合数
max_steps_per_episode = 100  # 每回合的最大步数

for episode in range(num_episodes):
    state = env.reset()  # 重置环境
    done = False
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        # 根据当前状态选择动作
        action = agent.choose_action(state)
        # 对于较新的Gym版本，你需要捕获 terminated 和 truncated
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # 两者只要有一个为True，就表示回合结束

        # 更新Q值表
        agent.update_q_table(state, action, reward, next_state, done)

        # 更新状态
        state = next_state
        total_reward += reward

        if done:
            break

    # 打印当前回合的结果
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

# 测试Q-Learning智能体
num_test_episodes = 10

for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    print(f"\nTest Episode {episode + 1}")
    
    while not done:
        env.render()  # 渲染当前环境
        action = np.argmax(agent.q_table[state])  # 选择最优动作
        state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Test Episode Reward: {total_reward}")

env.close()
