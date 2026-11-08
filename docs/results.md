# Final agent

The best experiment was the baseline (learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 to 0.02 over 100,000 steps). It was trained again for 800,000 steps (382 games, about 90 minutes on a Colab T4) and saved as [`dqn_model.pth`](../dqn_model.pth).

![Training plot of the final agent](../models/shima/best/training_plot.png)

| Training steps | Average reward (last 10 games) |
|---|---|
| 100,000 | -18.4 |
| 200,000 | -13.6 |
| 300,000 | -1.9 |
| 400,000 | +16.3 |
| 600,000 | +18.5 |
| 800,000 | +19.9 |

## What the episode length shows

1. For the first 50,000 steps or so, games last about 900 steps. The agent misses almost every ball and loses every point.
2. Between 50,000 and 300,000 steps, games get much longer and peak above 5,000 steps. The agent has learned to return the ball, so rallies last, but it rarely scores.
3. After 300,000 steps, games shorten to about 1,800 steps while the reward climbs to +20, because the agent now wins its points quickly. It won 41 of its training games 21-0.

[`models/shima/best/training_log.csv`](../models/shima/best/training_log.csv) has the episode length, total steps and reward of every game.

## Evaluation

`play.py` always picks the action with the highest Q-value and makes no random moves. It won all 3 recorded games 21-0: [game 1](../videos/pong-dqn-episode-0.mp4), [game 2](../videos/pong-dqn-episode-1.mp4), [game 3](../videos/pong-dqn-episode-2.mp4).
