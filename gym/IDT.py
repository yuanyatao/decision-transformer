import torch
import numpy as np
from decision_transformer.models.decision_transformer import DecisionTransformer
from src.iql import ImplicitQLearning
from experiment import discount_cumsum, evaluate_episode_rtg
from argparse import ArgumentParser
import pickle
import gym
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
class IDT:
    def __init__(
        self, 
        iql: ImplicitQLearning,
        dt: DecisionTransformer,
        env,
        dataset,
        K,
        device,
        max_ep_len,
        scale=1000.0
    ):
        """
        Integrates Implicit Q-Learning (IQL) and Decision Transformer (DT) for enhanced training.

        Args:
            iql: Pre-trained IQL model to provide Q and V values.
            dt: Decision Transformer model for sequence modeling.
            env: Environment object for evaluation.
            dataset: Offline RL dataset.
            K: Context length for DT.
            device: Device to use for training (cuda or cpu).
            max_ep_len: Maximum episode length.
            scale: Scaling factor for returns.
        """
        self.iql = iql
        self.dt = dt.to(device)
        self.env = env
        self.dataset = dataset
        self.K = K
        self.device = device
        self.max_ep_len = max_ep_len
        self.scale = scale
        
        # Precompute state normalization from dataset
        states = np.concatenate([traj['observations'] for traj in dataset], axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6

    def preprocess_data(self):
        """
        Preprocess dataset to replace returns with IQL Q values.
        """
        for traj in self.dataset:
            states = torch.tensor(traj['observations'], dtype=torch.float32, device=self.device)
            actions = torch.tensor(traj['actions'], dtype=torch.float32, device=self.device)
            traj['iql_q'] = self.iql.qf(states, actions).detach().cpu().numpy()

    def get_batch(self, batch_size):
        """
        Sample a batch of data for training DT.
        
        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Preprocessed batch ready for DT training.
        """
        batch_inds = np.random.choice(len(self.dataset), size=batch_size, replace=True)

        s, a, rtg, timesteps, mask = [], [], [], [], []
        for ind in batch_inds:
            traj = self.dataset[ind]
            traj_len = len(traj['observations'])
            start = np.random.randint(0, traj_len)
            
            s_traj = traj['observations'][start:start + self.K]
            a_traj = traj['actions'][start:start + self.K]
            q_traj = traj['iql_q'][start:start + self.K]

            # Normalize states and adjust sequence length
            tlen = len(s_traj)
            s_traj = (s_traj - self.state_mean) / self.state_std

            s.append(np.pad(s_traj, ((self.K - tlen, 0), (0, 0)), constant_values=0))
            a.append(np.pad(a_traj, ((self.K - tlen, 0), (0, 0)), constant_values=-10))
            rtg.append(np.pad(q_traj, (self.K - tlen, 0), constant_values=0))
            timesteps.append(np.pad(np.arange(start, start + tlen), (self.K - tlen, 0), constant_values=0))
            mask.append(np.pad(np.ones(tlen), (self.K - tlen, 0), constant_values=0))

        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.stack(a), dtype=torch.float32, device=self.device)
        rtg = torch.tensor(np.stack(rtg), dtype=torch.float32, device=self.device) / self.scale
        timesteps = torch.tensor(np.stack(timesteps), dtype=torch.long, device=self.device)
        mask = torch.tensor(np.stack(mask), dtype=torch.long, device=self.device)

        return s, a, rtg, timesteps, mask

    def train(self, optimizer, scheduler, batch_size, num_steps):
        """
        Train the DT model with preprocessed data.

        Args:
            optimizer: Optimizer for DT model.
            scheduler: Learning rate scheduler.
            batch_size: Number of samples in each batch.
            num_steps: Total number of training steps.
        """
        self.preprocess_data()

        for step in range(num_steps):
            s, a, rtg, timesteps, mask = self.get_batch(batch_size)
            
            optimizer.zero_grad()
            state_preds, action_preds, return_preds = self.dt(s, a, None, rtg, timesteps, attention_mask=mask)

            loss = torch.mean((action_preds - a) ** 2)  # MSE Loss for actions
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    def evaluate(self, num_episodes, target_return):
        """
        Evaluate the trained DT model.

        Args:
            num_episodes: Number of episodes to evaluate.
            target_return: Target return for evaluation.

        Returns:
            Mean return and length of episodes.
        """
        returns, lengths = [], []
        for _ in range(num_episodes):
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    env=self.env,
                    state_dim=self.dt.state_dim,
                    act_dim=self.dt.act_dim,
                    model=self.dt,
                    max_ep_len=self.max_ep_len,
                    scale=self.scale,
                    target_return=target_return / self.scale,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                )
            returns.append(ret)
            lengths.append(length)

        return np.mean(returns), np.mean(lengths)

# Example training script
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env-name', default='Hopper-v3')
    parser.add_argument('--log-dir', default='./logs/idt')
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
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--dataset-path', type=str, default='./data/hopper-medium-v2.pkl', help='Path to the dataset PKL file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    args = parser.parse_args()

    # Load environment
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load dataset
    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Initialize IQL components
    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(args.device),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(args.device),
        policy=GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(args.device),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
    )

    # Initialize DT model
    dt_model = DecisionTransformer(
        state_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_dim,
        max_length=20,
        max_ep_len=args.max_episode_steps,
        n_layer=3,
        n_head=4,
        n_inner=4 * args.hidden_dim,
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )

    idt = IDT(
        iql=iql,
        dt=dt_model,
        env=env,
        dataset=dataset,
        K=20,
        device=args.device,
        max_ep_len=args.max_episode_steps,
        scale=1000.0
    )

    optimizer = torch.optim.AdamW(dt_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/10000, 1))

    idt.train(optimizer=optimizer, scheduler=scheduler, batch_size=args.batch_size, num_steps=args.n_steps)

    mean_return, mean_length = idt.evaluate(num_episodes=10, target_return=3600)
    print(f"Evaluation Results - Mean Return: {mean_return}, Mean Length: {mean_length}")
