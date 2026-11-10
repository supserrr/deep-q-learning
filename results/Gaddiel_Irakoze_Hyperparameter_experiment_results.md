# Hyperparameter experiments: Gaddiel Irakoze

**Environment:** `ALE/Pong-v5` (Atari Pong)  
**Algorithm:** Deep Q-Network (DQN), implemented in `train.py` with PyTorch  
**Training budget:** 200,000 environment steps per experiment; seed 42  
**Comparison:** Each experiment uses the same DQN architecture and fixed training settings, but combines changes to the learning rate, discount factor, batch size and/or epsilon schedule.

A reward of +1 is earned for each point won and -1 for each point lost. A complete Pong episode ends with a reward between -21 and +21. The *final average reward* below is the arithmetic mean over the final ten completed **training** episodes (which still use epsilon-greedy actions), not a separate deterministic evaluation. Longer episodes can indicate that the paddle is returning the ball more often, but episode length alone is not proof of good scoring performance.

## Results of the ten experiments

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| **1** | **0.0001** | **0.99** | **64** | **1.0** | **0.01** | **150,000** | **Best.** Final average reward **+12.0**; best individual episode **+18**. The first positive-reward game occurred at step **158,238**. The last ten episodes averaged **2,451 steps**, compared with about **897** at the beginning; the agent progressed from frequent losses to returning shots and winning points. |
| 2 | 0.0002 | 0.99 | 64 | 1.0 | 0.02 | 100,000 | Final average **-20.8** (best **-18**), with no winning game. Last-ten episode length **810 steps**: the agent continued losing points quickly. Increasing learning rate to 0.0002 together with batch size 64 did not produce learning in this run. |
| **3** | **0.0003** | **0.99** | **128** | **1.0** | **0.02** | **100,000** | **Worst.** Final average **-21.0** (best **-17**), with no winning game. Last-ten games averaged only **801 steps**. This combination of a higher learning rate and very large batch made no measurable improvement within the training budget. |
| 4 | 0.00005 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Final average **-13.5** (best **-5**), no winning game. Last-ten games averaged **3,505 steps**, versus about **967** initially. The paddle appears to have learned to sustain longer rallies, but scoring performance remained weak. The lower learning rate may need more training steps. |
| 5 | 0.001 | 0.99 | 32 | 1.0 | 0.02 | 100,000 | Final average **-20.9** (best **-18**), no winning game, average ending episode length **785 steps**. The much higher learning rate was associated with virtually no learning. Overly large updates are a plausible explanation, but Q-value instability was not directly measured. |
| 6 | 0.0002 | 0.97 | 32 | 1.0 | 0.02 | 100,000 | Final average **-20.9** (best **-18**), no winning game, average ending episode length **820 steps**. Higher learning rate combined with lower gamma performed poorly; because both changed, the experiment cannot isolate which contributed most. |
| 7 | 0.0001 | 0.995 | 32 | 1.0 | 0.02 | 150,000 | Final average **-13.5** (best **-10**), no winning game. Last-ten games averaged **2,906 steps**; the agent learned to keep rallies going longer but did not win a game. A higher gamma and slower exploration decay may require additional steps to translate defense into scoring. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Final average **-14.7** (best **-9**), no winning game. Last-ten games averaged **3,040 steps**. Fast exploration decay and a higher final epsilon produced some ability to return the ball, but continuing random actions could still disrupt precise shots. |
| 9 | 0.0001 | 0.99 | 32 | 0.8 | 0.01 | 200,000 | Final average **-14.3** (best **-11**), no winning game. Last-ten games averaged **2,515 steps**. Lower initial randomness and a slower epsilon decay led to partial improvement, but not consistent scoring by step 200,000. |
| 10 | 0.0002 | 0.99 | 32 | 1.0 | 0.01 | 60,000 | Final average **-20.9** (best **-18**), no winning game. Last-ten episode length **803 steps**. The higher learning rate together with rapid exploration decay did not yield useful play in this run. |

The model weights, full episode logs, plots, and exact configurations are stored in `models/gaddiel/exp01/` through `models/gaddiel/exp10/`. The machine-generated numerical table is `models/gaddiel/results_table.md`.

## Best and worst configurations

**Best — Experiment 1.** With learning rate 0.0001, gamma 0.99, batch size 64, epsilon 1.0 → 0.01 over 150,000 steps, the final ten training games averaged **+12.0**, and the best single game scored **+18**. It was the only one of my ten runs to finish an episode with a positive reward (14 positive-reward episodes in its full training log). This is the configuration I would choose from my 200,000-step runs.

**Worst — Experiment 3.** Learning rate 0.0003 combined with batch size 128 ended at **-21.0** on average. Experiments 2, 5, 6 and 10 also stayed close to -21, but Experiment 3 had the lowest measured final average. The short episodes and lack of positive rewards are consistent with an agent that did not learn to defend effectively within 200,000 steps.

## What the combinations taught me

1. **Learning rate was particularly influential in this set of runs.** Every experiment with learning rate 0.0002 or higher (2, 3, 5, 6 and 10) finished between -20.8 and -21.0. Learning rate 0.0001 worked better in several configurations, particularly Experiment 1. Large updates *may* make DQN learning unstable; these training logs alone do not establish the mechanism.
2. **Longer rallies do not necessarily mean more points won.** Experiments 4, 7, 8 and 9 extended their episode lengths substantially but still finished with negative average rewards. Defending the paddle is only part of winning Pong.
3. **Combined changes can have a different outcome from changing one parameter alone.** Experiment 1 combined batch size 64 with a lower final epsilon and a longer decay period and outperformed my other configurations. Because I varied several parameters simultaneously, I cannot attribute the entire gain to any one of those changes.
4. **The training budget matters.** Runs that were still improving near 200,000 steps might perform differently with further training. The ranking describes this particular budget and seed, not a universal optimum.

## Comparison with Shima's experiments

Shima's baseline (Experiment 1: learning rate 0.0001, gamma 0.99, batch 32, epsilon 1.0 → 0.02 over 100,000 steps) ended at a **-8.0** average. My Experiment 1 ended at **+12.0**, a **20-point higher** last-ten-episode training average at the same 200,000-step budget. Shima's Experiment 6 tested batch size 64 alone and ended at **-9.0**; my Experiment 1 also used batch size 64, but with **epsilon end 0.01** and **150,000-step decay**, so the combination cannot be reduced to a batch-size effect.

The poor results with higher learning rates are consistent across the two sets: Shima's learning-rate-0.0005 experiment ended at **-20.9**, and all of my runs at learning rate 0.0002 or above also ended near -21. Shima's lower-learning-rate (0.000025) experiment reached **-16.2**, while my learning-rate-0.00005 run reached **-13.5**, though other settings differed.

The comparison is **descriptive, not a controlled causal study**: every configuration used one seed, some configurations changed multiple parameters, and training reward was measured under different epsilon schedules. A fairer final policy comparison would run the saved models greedily for multiple evaluation episodes under an identical procedure.

## Optional longer training run (not part of the ten-run comparison)

The notebook also retrained my best configuration for **800,000 steps**, saving the separate checkpoint and log under `models/gaddiel/best/` in Google Drive. Its last ten completed **training** episodes averaged **+18.0**, versus **+12.0** after the 200,000-step Experiment 1. The group repository's root `dqn_model.pth` remains Shima's final agent; my longer-run result is reported here as an optional extension, not a replacement or a verified head-to-head evaluation.
