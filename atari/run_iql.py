import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy,DeterministicPolicy,DiscretePolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import set_seed, Log
from mingpt.trainer_atari import Env, Args
from create_dataset import create_dataset

# 自定义数据集类
class StateActionDataset(torch.utils.data.Dataset):
    def __init__(self, data, next_data, actions, rewards, terminals):
        """
        数据集类，用于加载状态、动作、奖励和终止标志。
        :param data: 当前状态，形状为 (num_samples, channels, height, width)
        :param next_data: 下一状态，形状与 data 一致
        :param actions: 动作数据，形状为 (num_samples,)
        :param rewards: 奖励数据，形状为 (num_samples,)
        :param terminals: 终止标志，形状为 (num_samples,)
        """
        self.data = data
        self.next_data = next_data
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :return: 当前状态、下一状态、动作、奖励和终止标志
        """
        state = torch.tensor(self.data[idx], dtype=torch.float32) / 255.0
        next_state = torch.tensor(self.next_data[idx], dtype=torch.float32) / 255.0
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        terminal = torch.tensor(self.terminals[idx], dtype=torch.float32)

        return state, next_state, action, reward, terminal

def main(args):
    # 设置随机种子以确保结果可复现
    set_seed(args.seed)

    # 创建日志目录并记录参数
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path(args.log_dir) / args.game / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log = Log(log_dir, vars(args))
    log(f'Log dir: {log.dir}')

    # 加载数据集
    print("Loading dataset...")
    obss, next_obss, actions, rewards, terminals = create_dataset(
        num_buffers=args.num_buffers,
        num_steps=args.num_steps,
        game=args.game,
        data_dir_prefix=args.data_dir_prefix,
        trajectories_per_buffer=args.trajectories_per_buffer,
        mode="IQL"  # 以 IQL 模式加载数据
    )
    # print()
    # print('obss.shape:',obss.shape)
    print(f"Loaded raw dataset: {len(obss)} observations")

    # 创建数据集和 DataLoader
    train_dataset = StateActionDataset(obss, next_obss, actions, rewards, terminals)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 确定状态通道数和动作维度
    state_channels = 4  # Atari游戏通常有4帧叠加
    act_dim = max(actions) + 1 
    # print('act_dim:',act_dim)
    # 初始化策略网络和 IQL 模型
    # policy = GaussianPolicy(state_channels, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    # policy = DeterministicPolicy(state_channels, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    policy = DiscretePolicy(state_channels, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    iql = ImplicitQLearning(
        qf=TwinQ(state_channels, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(state_channels, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    # 定义策略评估函数
    def eval_policy():
        try:
            env = Env(Args(args.game.lower(), args.seed))
            env.eval()
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            return 0.0, 0.0

        eval_returns = []

        for _ in range(args.n_eval_episodes):
            total_reward = 0
            state = env.reset().to(args.device).unsqueeze(0)  # 这里不再需要归一化了 Env里面自动归一化了
            done = False

            while not done:
                with torch.no_grad():
                    action = policy.act(state)[0]

                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state.to(args.device).unsqueeze(0) # 这里也不应该/255.

            eval_returns.append(total_reward)

        env.close()
        mean_return = np.mean(eval_returns)
        std_return = np.std(eval_returns)
        log.row({'return mean': mean_return, 'return std': std_return})

        return mean_return, std_return

    # 开始训练
    best_return = -float('inf')
    # try:
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}...")

        # 设置为训练模式
        iql.qf.train()
        iql.vf.train()
        iql.policy.train()

        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for states, next_states, actions, rewards, terminals in pbar:
            # 将数据移动到设备并归一化
            states = states.to(args.device)
            next_states = next_states.to(args.device)
            actions = actions.to(args.device)
            rewards = rewards.to(args.device)
            terminals = terminals.to(args.device)

            # 调用 IQL 更新方法
            total_loss, losses = iql.update(states, next_states, actions, rewards, terminals)
            epoch_loss += total_loss
            pbar.set_postfix(loss=total_loss)

        print(f"Epoch {epoch + 1} training loss: {epoch_loss / len(train_dataloader):.4f}")

        # 验证策略性能
        iql.qf.eval()
        iql.vf.eval()
        iql.policy.eval()
        mean_return, std_return = eval_policy()
        best_return = max(best_return, mean_return)
        print(f"Epoch {epoch + 1} mean return: {mean_return:.2f}, std return: {std_return:.2f}, best return: {best_return:.2f}")

    # except RuntimeError as e:
        # print(f"Runtime error during training: {e}")
        # return None

    # 保存模型
    model_save_dir = Path(args.log_dir) / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / f'iql_{args.game}_{timestamp}.pt'
    torch.save(iql.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    parser.add_argument('--context_length', type=int, default=30, help='上下文长度，用于序列建模')
    parser.add_argument('--epochs', type=int, default=5, help='训练的轮数')
    parser.add_argument('--num_steps', type=int, default=500000, help='训练中使用的步数')
    parser.add_argument('--num_buffers', type=int, default=50, help='数据缓冲区的数量')
    parser.add_argument('--game', type=str, default='Breakout', help='游戏名称')
    parser.add_argument('--batch_size', type=int, default=128, help='训练的批次大小')
    parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='每个缓冲区的轨迹数')
    parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/', help='数据目录的前缀')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_hidden', type=int, default=2, help='隐藏层数量')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--alpha', type=float, default=0.005, help='Q函数更新的学习率')
    parser.add_argument('--tau', type=float, default=0.7, help='IQL的超参数tau')
    parser.add_argument('--beta', type=float, default=3.0, help='IQL的超参数beta')
    parser.add_argument('--eval_period', type=int, default=1, help='评估间隔（以 epoch 为单位）')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='评估轮数')
    parser.add_argument('--log_dir', type=str, default='./logs/iql', help='日志目录')
    parser.add_argument('--discount', type=float, default=0.99, help='折扣因子')
    args = parser.parse_args()

    main(args)
