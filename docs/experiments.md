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

To be added after Gaddiel's runs: how the combined changes compare with the single-change results.
