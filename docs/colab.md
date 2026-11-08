# Training on Colab

All training ran on a free Colab T4 GPU with [`notebooks/train_on_colab.ipynb`](../notebooks/train_on_colab.ipynb). The notebook clones this repo into Google Drive, installs the packages with uv, and runs `tools/experiments.py`.

## Steps

1. Open the notebook in Colab: [train_on_colab.ipynb](https://colab.research.google.com/github/supserrr/deep-q-learning/blob/main/notebooks/train_on_colab.ipynb).
2. Runtime > Change runtime type > T4 GPU.
3. Set `MEMBER` in the first code cell (`"shima"` or `"gaddiel"`).
4. Runtime > Run all, then allow access to Google Drive.

The notebook runs the member's 10 experiments two at a time, shows the results table and plots, then retrains the best experiment for 800,000 steps and copies it to `dqn_model.pth`.

## Timing

| Run | Time on a T4 |
|---|---|
| One pair of experiments (200,000 steps each) | about 33 minutes |
| All 10 experiments | about 3 hours |
| Final run (800,000 steps) | about 90 minutes |

## Notes

- The project lives in `MyDrive/deep-q-learning`, so a disconnect loses nothing. Run all cells again and finished experiments are skipped.
- The uv environment goes on Colab's local disk (`/content/.venv`), which is much faster than Drive.
- `MPLBACKEND=Agg` is set because Colab's default plot backend is not installed in the uv environment.
- To bring results back, download the `models` folder from Google Drive and add it to the repo through a pull request.
