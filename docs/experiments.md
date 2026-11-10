# Hyperparameter experiments

Each member ran 10 experiments of 200,000 steps with seed 42, tuning the learning rate, gamma, batch size and epsilon schedule. Both sets start from the same baseline, so all 20 runs compare directly.

| Member | Approach | Results |
|---|---|---|
| Shima Serein | One hyperparameter changed at a time, to see each one's effect alone | [Shima_Serein_Hyperparameter_experiment_results.md](../results/Shima_Serein_Hyperparameter_experiment_results.md) |
| Gaddiel Irakoze | Several hyperparameters changed together, to look for a better combination | [Gaddiel_Irakoze_Hyperparameter_experiment_results.md](../results/Gaddiel_Irakoze_Hyperparameter_experiment_results.md) |

Baseline: learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 to 0.02 over 100,000 steps.

## Values tested

| Hyperparameter | Shima | Gaddiel |
|---|---|---|
| Learning rate | 0.000025, 0.0001, 0.0005 | 0.00005, 0.0001, 0.0002, 0.0003, 0.001 |
| Gamma | 0.90, 0.95, 0.99 | 0.97, 0.99, 0.995 |
| Batch size | 32, 64 | 32, 64, 128 |
| Epsilon start | 0.5, 1.0 | 0.8, 1.0 |
| Epsilon end | 0.02, 0.10 | 0.01, 0.02, 0.05 |
| Epsilon decay (steps) | 30,000, 100,000, 250,000 | 50,000, 60,000, 100,000, 150,000, 200,000 |

The exact configurations are in [`tools/experiments.py`](../tools/experiments.py).

## Outputs

Each run saves to `models/<member>/expNN/`:

| File | Content |
|---|---|
| `dqn_model.pth` | trained online network weights |
| `training_log.csv` | one row per game: episode length, total steps, reward, epsilon |
| `training_plot.png` | reward and episode length over training |
| `config.json` | the hyperparameters used |

`models/<member>/results_table.md` is a generated table of each run's numbers. Each member writes their observations in their own file in `results/`.

## Running them

```bash
uv run tools/experiments.py --member shima --parallel 2                 # all 10, two at a time
uv run tools/experiments.py --member shima --only 3                     # one experiment
uv run tools/experiments.py --member shima --table                      # rebuild the numbers table
uv run tools/experiments.py --member shima --table --best-steps 800000  # retrain the best one for longer
```

Finished runs are skipped, so an interrupted session can simply be restarted. We trained on Colab: see [colab.md](colab.md).

## Comparison

At the same 200,000-step training budget, **Gaddiel's Experiment 1** (learning rate 0.0001, gamma 0.99, batch 64, epsilon 1.0 → 0.01 over 150,000 steps) reached **+12.0** average reward over its final ten training episodes. This was higher than **Shima's baseline Experiment 1** (**-8.0**) and Shima's **batch-64-only Experiment 6** (**-9.0**). The 20-point gap against Shima's baseline suggests this combined configuration was more successful in these runs, but batch size, final epsilon and decay changed together, so the effect of each parameter cannot be isolated.

Both members' results suggest that **larger learning rates performed poorly** with this network and 200,000-step budget. Shima's learning rate 0.0005 experiment finished at **-20.9**; Gaddiel's experiments with learning rates 0.0002, 0.0003 and 0.001 ended between **-20.8 and -21.0**. In contrast, lower-learning-rate runs were generally able to sustain longer rallies even when they did not win games. Gaddiel's Experiment 4 (learning rate 0.00005) averaged **-13.5** with approximately **3,505 steps** per final episode, while Shima's learning rate 0.000025 experiment averaged **-16.2**.

Gaddiel's other combinations involving gamma or changes to epsilon usually improved episode length without achieving positive average rewards. This reinforces the distinction between **keeping the ball in play** and **scoring enough points to win**. These are single-seed results, and the training averages were collected under different epsilon-greedy policies. The comparisons therefore describe observed learning in these experiments; choosing a final agent with greater confidence would require several greedy evaluation games for each saved model under identical conditions. Shima's existing root-level `dqn_model.pth` and recorded demonstration remain unchanged.
