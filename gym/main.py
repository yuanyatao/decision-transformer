from pathlib import Path
import gym
import numpy as np
import torch
from tqdm import trange
import pickle

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import Log, set_seed as base_set_seed, evaluate_policy


def set_seed(seed, env=None):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env:
        env.reset(seed=seed)


def load_dataset_from_pkl(dataset_path):
    """Load dataset from a PKL file."""
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    states, actions, rewards, terminals = [], [], [], []
    for traj in trajectories:
        states.append(traj['observations'])
        actions.append(traj['actions'])
        rewards.append(traj['rewards'])
        terminals.append(traj['terminals'] if 'terminals' in traj else traj['dones'])

    return {
        'observations': np.concatenate(states, axis=0),
        'actions': np.concatenate(actions, axis=0),
        'rewards': np.concatenate(rewards, axis=0),
        'terminals': np.concatenate(terminals, axis=0),
    }


def preprocess_dataset(dataset, max_episode_steps):
    """Normalize rewards and convert dataset to PyTorch tensors."""
    min_ret, max_ret = np.min(dataset['rewards']), np.max(dataset['rewards'])
    dataset['rewards'] /= (max_ret - min_ret)
    dataset['rewards'] *= max_episode_steps

    for key in dataset:
        dataset[key] = torch.tensor(dataset[key], dtype=torch.float32)
    return dataset


def sample_batch(dataset, batch_size, device):
    """Sample a batch of data from the dataset."""
    indices = np.random.choice(len(dataset['observations']) - 1, size=batch_size, replace=False)
    batch = {
        'observations': dataset['observations'][indices].to(device),
        'actions': dataset['actions'][indices].to(device),
        'rewards': dataset['rewards'][indices].to(device),
        'terminals': dataset['terminals'][indices].to(device),
        'next_observations': dataset['observations'][indices + 1].to(device),
    }
    return batch


def get_env_and_dataset(log, env_name, max_episode_steps, dataset_path=None):
    """Load the environment and dataset."""
    env = gym.make(env_name)
    if dataset_path:
        log(f"Loading dataset from {dataset_path}")
        dataset = load_dataset_from_pkl(dataset_path)
    else:
        raise ValueError("Dataset path must be specified for loading PKL data.")
    dataset = preprocess_dataset(dataset, max_episode_steps)
    return env, dataset


def main(args):
    """Main function for training Implicit Q-Learning."""
    torch.set_num_threads(1)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    log = Log(Path(args.log_dir) / args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps, args.dataset_path)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device)

    def eval_policy():
        """Evaluate the policy."""
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) for _ in range(args.n_eval_episodes)])
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(device),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
    )

    for step in trange(args.n_steps):
        batch = sample_batch(dataset, args.batch_size, device)
        iql.update(
            observations=batch['observations'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            next_observations=batch['next_observations'],
        )
        if (step + 1) % args.eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), log.dir / 'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--env-name', default='Hopper-v3')
    parser.add_argument('--log-dir', default='./logs/iql')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--dataset-path', type=str,default='./data/hopper-medium-v2.pkl', help='Path to the dataset PKL file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    main(parser.parse_args())
