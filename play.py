"""Task 4: load the trained DQN and watch it play Pong in real time.

    uv run play.py                                   # plays with dqn_model.pth in a window
    uv run play.py --model models/shima/exp01/dqn_model.pth --episodes 3
    uv run play.py --video videos                    # save an .mp4 per game instead of opening a window
"""

import argparse

import numpy as np
import torch

from tools.retro_view import RetroView
from train import QNetwork, make_env  # same environment and network as in training


def main():
    p = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong")
    p.add_argument("--model", default="dqn_model.pth")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--video", metavar="FOLDER", help="save videos to this folder instead of showing a window")
    args = p.parse_args()

    # 1. Same Atari environment as in training. RetroView draws the game in black and white;
    #    the agent still sees the original Atari frames.
    env = make_env(view=lambda game: RetroView(game, video_folder=args.video))

    # 2-4. Rebuild the same network and load the trained weights.
    model = QNetwork(env.action_space.n)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()  # 5. evaluation mode: no training, no exploration

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            # 6. Greedy action: always the highest predicted Q-value.
            with torch.no_grad():
                q_values = model(torch.as_tensor(np.asarray(state)).unsqueeze(0))
            action = int(q_values.argmax(dim=1).item())
            # 7. Each step plays 4 Atari frames, and RetroView draws every one of them.
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode}: reward {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    main()
