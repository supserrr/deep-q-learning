"""Task 2: train a Deep Q-Network (DQN) from scratch to play Atari Pong.

Example:
    uv run train.py --member shima --name exp01 --lr 1e-4 --gamma 0.99 --batch-size 32 \
        --eps-start 1.0 --eps-end 0.02 --eps-decay 100000 --steps 300000

Everything for one run is saved in models/<member>/<name>/:
    dqn_model.pth      trained online network weights
    training_log.csv   one row per episode: length, total steps so far, reward
    training_plot.png  reward and episode length over training
    config.json        the hyperparameters used
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed (works on Colab too)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"


def make_env(view=None):
    """Create Pong with the standard DQN preprocessing.

    - Each action is repeated for 4 frames (the agent decides 15 times per second).
    - Frames are converted to 84x84 grayscale.
    - The last 4 frames are stacked so the network can see the ball's direction and speed.

    view: optional wrapper that draws the raw game (play.py uses it to show the game).
    """
    # frameskip=1 here because AtariPreprocessing does the frame skipping itself.
    # repeat_action_probability=0 turns off "sticky actions", as in the original DQN paper.
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0)
    if view:
        env = view(env)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    """CNN from the DQN Nature paper: 3 conv layers, then 2 fully connected layers.

    Input: 4 stacked 84x84 frames. Output: one Q-value per action.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0  # pixels 0-255 -> 0-1
        return self.fc(self.conv(x))


class ReplayBuffer:
    """Fixed-size memory of past transitions. When full, the oldest ones are overwritten."""

    def __init__(self, capacity, obs_shape):
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.position = 0  # where the next transition is written
        self.size = 0  # how many transitions are stored

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Pick random transitions. Random sampling breaks the correlation between consecutive frames."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def epsilon_at(step, eps_start, eps_end, eps_decay):
    """Linear epsilon decay: from eps_start to eps_end over eps_decay steps, then stay at eps_end."""
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(network, state, epsilon, n_actions, device):
    """Epsilon-greedy: random action with probability epsilon, otherwise the action with the highest Q-value."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q_values = network(torch.as_tensor(np.asarray(state), device=device).unsqueeze(0))
    return int(q_values.argmax(dim=1).item())


def learn(online, target, optimizer, batch, gamma):
    """One gradient step on a batch of transitions. Returns the loss."""
    states, actions, rewards, next_states, dones = batch

    # Q(s, a) from the online network, for the actions that were actually taken.
    q_values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + gamma * max_a' Q_target(s', a').
    # The target network is a frozen copy, so the target does not move every step.
    # When the game is over (done = 1) there is no future reward.
    with torch.no_grad():
        next_q_values = target(next_states).max(dim=1).values
        targets = rewards + gamma * next_q_values * (1 - dones)

    # Huber loss: like squared error for small errors, but grows linearly for big ones (more stable).
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10)  # avoid huge updates
    optimizer.step()
    return loss


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # Colab GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def save_plot(episodes, path, title):
    """Plot reward and episode length against the number of training steps."""
    steps = [e["total_steps"] for e in episodes]
    rewards = [e["reward"] for e in episodes]
    lengths = [e["episode_length"] for e in episodes]
    window = 10
    moving_avg = [np.mean(rewards[max(0, i - window + 1) : i + 1]) for i in range(len(rewards))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(steps, rewards, alpha=0.35, label="episode reward")
    ax1.plot(steps, moving_avg, linewidth=2, label=f"average of last {window}")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel("Reward (points won - points lost)")
    ax1.legend(loc="upper left")
    ax2.plot(steps, lengths, color="tab:green")
    ax2.set_ylabel("Episode length (steps)")
    ax2.set_xlabel("Training steps")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args):
    out_dir = Path("models") / args.member / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Training on {device}, saving to {out_dir}")

    env = make_env()
    n_actions = env.action_space.n

    # Online network: the one we train. Target network: a periodically updated copy used for the Bellman target.
    online = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape)

    episodes = []  # one dict per finished episode
    state, _ = env.reset(seed=args.seed)
    episode_reward, episode_length = 0.0, 0
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # 1. Choose an action (epsilon-greedy).
        epsilon = epsilon_at(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(online, state, epsilon, n_actions, device)

        # 2. Play it and store the experience. Rewards are clipped to -1, 0 or +1 (sign) for stable learning.
        next_state, reward, terminated, truncated, _ = env.step(action)
        buffer.add(state, action, np.sign(reward), next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if terminated or truncated:
            episodes.append({"episode": len(episodes) + 1, "episode_length": episode_length,
                             "total_steps": step, "reward": episode_reward, "epsilon": round(epsilon, 4)})
            recent = np.mean([e["reward"] for e in episodes[-10:]])
            speed = step / (time.time() - start_time)
            print(f"{args.name} | step {step:>8} | episode {len(episodes):>4} | reward {episode_reward:>5.0f} | "
                  f"avg10 {recent:>6.1f} | length {episode_length:>5} | eps {epsilon:.3f} | {speed:.0f} steps/s",
                  flush=True)
            state, _ = env.reset()
            episode_reward, episode_length = 0.0, 0

        # 3. Learn from a random batch of past experiences (once the buffer has enough).
        if step > args.learning_starts and step % args.train_freq == 0:
            learn(online, target, optimizer, buffer.sample(args.batch_size, device), args.gamma)

        # 4. Copy the online network into the target network every few thousand steps.
        if step % args.target_update == 0:
            target.load_state_dict(online.state_dict())

        # Save regularly so a disconnected Colab session does not lose everything.
        if step % 50_000 == 0:
            torch.save(online.state_dict(), out_dir / "dqn_model.pth")

    env.close()
    torch.save(online.state_dict(), out_dir / "dqn_model.pth")

    with open(out_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    save_plot(episodes, out_dir / "training_plot.png", f"{args.member} / {args.name}")
    minutes = (time.time() - start_time) / 60
    print(f"Done in {minutes:.1f} min. Average reward of last 10 episodes: "
          f"{np.mean([e['reward'] for e in episodes[-10:]]):.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on Atari Pong")
    p.add_argument("--member", default="shima", help="folder name for this group member")
    p.add_argument("--name", default="exp01", help="experiment name")
    # The hyperparameters each member tunes (Task 3)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay", type=int, default=100_000, help="steps to go from eps-start to eps-end")
    # Fixed settings, the same for every experiment
    p.add_argument("--steps", type=int, default=300_000, help="total environment steps")
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=10_000, help="random play before learning starts")
    p.add_argument("--train-freq", type=int, default=1, help="learn every N steps")
    p.add_argument("--target-update", type=int, default=1_000, help="copy online -> target every N steps")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
