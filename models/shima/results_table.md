# Experiment results (generated): Shima Serein

Environment: ALE/Pong-v5. Every experiment trains for 200,000 steps with the same seed.
Epsilon decays linearly from start to end over the number of steps in the decay column.
Reward is points won minus points lost in one game (-21 to +21).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -8.0 (best 3); first won a game at step 198,386; games lasted 3166 steps at the end. |
| 2 | 0.0005 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -20.9 (best -17); never won a game; games lasted 814 steps at the end. |
| 3 | 2.5e-05 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -16.2 (best -13); never won a game; games lasted 3135 steps at the end. |
| 4 | 0.0001 | 0.95 | 32 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -12.8 (best -5); never won a game; games lasted 3163 steps at the end. |
| 5 | 0.0001 | 0.9 | 32 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -11.6 (best -5); never won a game; games lasted 3688 steps at the end. |
| 6 | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 100,000 | Avg reward of last 10 games: -9.0 (best -2); never won a game; games lasted 2982 steps at the end. |
| 7 | 0.0001 | 0.99 | 32 | 0.5 | 0.02 | 100,000 | Avg reward of last 10 games: -13.2 (best -5); never won a game; games lasted 3045 steps at the end. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Avg reward of last 10 games: -11.5 (best -2); never won a game; games lasted 2517 steps at the end. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 30,000 | Avg reward of last 10 games: -10.0 (best 1); first won a game at step 193,769; games lasted 2731 steps at the end. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.02 | 250,000 | Avg reward of last 10 games: -15.0 (best -11); never won a game; games lasted 2134 steps at the end. |
