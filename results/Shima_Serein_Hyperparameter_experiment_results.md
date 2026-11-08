# Hyperparameter experiments: Shima Serein

Environment: ALE/Pong-v5. Every experiment trains for 200,000 steps with the same seed.
Epsilon decays linearly from start to end over the number of steps in the decay column.
Reward is points won minus points lost in one game (-21 to +21). "Average reward" is the mean of the last 10 games.

I started from a baseline and changed one hyperparameter at a time, so each result shows the effect of that hyperparameter alone. Models, logs and plots are in `models/shima/expNN/`.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Best. Baseline. Average reward rose from -21 to -8.0 and it won its first game (+3) near the end. Games grew from ~900 to over 3,000 steps: it learned to return the ball first, then started to score. Still improving when training stopped. |
| 2 | 0.0005 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Worst. Never learned (-20.9, same as random). Games stayed short (~800 steps), so it never even learned to return the ball. The learning rate is too high: each update overshoots, and the Q-values chase a target that keeps moving. |
| 3 | 0.000025 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Learned, but slowly: -16.2 at the end, against -8.0 for the baseline. It only got past -18 at 155k steps (baseline: 78k). Long games (3,100 steps) show it learned to defend but not yet to score. The learning rate is too small for this training budget. |
| 4 | 0.0001 | 0.95 | 32 | 1.0 | 0.02 | 100,000 | -12.8. Started learning as fast as the baseline (past -18 at 77k) but levelled off lower. With gamma 0.95 a point 60 steps away is worth only ~5% of its value, so the agent cares less about setting up shots that score later. |
| 5 | 0.0001 | 0.90 | 32 | 1.0 | 0.02 | 100,000 | -11.6. Very short-sighted (a reward 20 steps away counts ~12%). It learned to keep the ball in play (longest games of all, up to 5,000 steps) but scored less than the baseline. Close to gamma 0.95; one seed cannot separate them. |
| 6 | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 100,000 | -9.0, second best (best game -2). Bigger batches give smoother gradient estimates, and it learned about as fast as the baseline, but each update costs more and it was no better per step. |
| 7 | 0.0001 | 0.99 | 32 | 0.5 | 0.02 | 100,000 | -13.2. Starting with only 50% random moves made the early buffer less varied. It began like the baseline but stalled around -13: with less exploration it saw fewer useful situations to learn from. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 100,000 | -11.5. Keeping 10% random moves forever caps performance: in Pong one random move at the wrong moment loses the point. Learning speed was fine, but its games were shorter than the baseline's. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 30,000 | -10.0, and it won a game (+1). Exploration ended early, so it started exploiting what it knew sooner: fastest to pass -12 (134k steps vs 176k for the baseline). Slightly behind the baseline at the end, probably because it explored less. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 250,000 | -15.0. Epsilon was still about 0.22 when training stopped, so roughly 1 in 5 moves was random. It learned slowly and the random moves kept costing points. Would need a much longer run. |

## Summary

The learning rate had the biggest effect: 0.0005 never learned, 0.000025 learned very slowly, and 0.0001 was clearly best. Gamma 0.99 beat 0.95 and 0.90, because Pong rewards arrive many steps after the shot that earns them and the agent has to value the future. The exploration schedule mattered less, but too much exploration late in training (epsilon end 0.10, or decay over 250k steps) cost points, while ending exploration early (30k) learned fastest at first. Batch size 64 was close to 32 and not worth the extra compute per step.

For the final agent I trained the baseline (experiment 1) for 800,000 steps. It averaged +19.9 over its last 10 games and won 3 of 3 evaluation games 21-0 with `play.py`.

Each experiment is a single run with seed 42, so differences of 1 to 2 points between runs can be noise.
