# Part A: SAM vs SGD on CIFAR-10

This repository contains the Part A experiment for comparing standard SGD and Sharpness-Aware Minimization (SAM-SGD) on CIFAR-10 in-distribution accuracy and simple out-of-distribution robustness.

The Part A goal is to establish the performance phenomenon clearly:

- train SGD and SAM-SGD from the same initialization;
- evaluate in-distribution CIFAR-10 test accuracy;
- evaluate OOD robustness under Gaussian noise, blur, and brightness shifts;
- save `last`, `best_id`, and `best_ood` checkpoints for each optimizer;
- generate report-ready CSV tables and figures.

The OOD results should be discussed per corruption, not only by average OOD accuracy, because SAM can improve blur and brightness robustness while behaving differently under noise.

## Project Structure

```text
.
|-- config.py
|-- data.py
|-- models.py
|-- sam.py
|-- train.py
|-- evaluate.py
|-- run_experiment.py
|-- plot_results.py
|-- utils.py
|-- requirements.txt
|-- Run_PartA_Colab.ipynb
|-- tests/
|-- data/
|-- checkpoints/
|-- results/
|-- figures/
`-- logs/
```

## Colab Workflow

Use `Run_PartA_Colab.ipynb` in Google Colab with GPU enabled.

Recommended runtime:

```text
Runtime > Change runtime type > GPU
```

The notebook will:

1. Mount Google Drive.
2. Install dependencies.
3. Run `python run_experiment.py`.
4. Run `python plot_results.py`.
5. Inspect generated checkpoints, CSV files, and figures.
6. Optionally load one selected checkpoint to verify it can be reused.

## Local Setup

```bash
pip install -r requirements.txt
```

## Run Experiment

```bash
python run_experiment.py
```

The experiment evaluates ID and OOD metrics at every epoch. It saves three checkpoint strategies for each optimizer:

- `last`: final training epoch.
- `best_id`: epoch with the highest ID accuracy.
- `best_ood`: epoch with the highest average OOD accuracy.

## Generate Figures and Summary Table

```bash
python plot_results.py
```

The plotting script reads `results/metrics.csv` and the per-epoch training logs, then writes the Part A figures and summary table.

## Results Files

The `results/` directory should contain lightweight CSV outputs:

```text
results/metrics.csv
results/training_log_sgd.csv
results/training_log_sam.csv
results/summary_table.csv
```

The updated `metrics.csv` stores one row per optimizer/checkpoint strategy:

```text
optimizer
checkpoint_type
epoch
train_loss
train_acc
id_loss
id_acc
ood_noise_loss
ood_noise_acc
ood_blur_loss
ood_blur_acc
ood_brightness_loss
ood_brightness_acc
avg_ood_acc
id_ood_acc_drop
```

The updated training logs store one row per epoch with train, ID, and OOD metrics. These logs support trajectory plots and justify separating `last`, `best_id`, and `best_ood`.

## Expected Checkpoints

```text
checkpoints/sgd_resnet18_cifar10_last.pt
checkpoints/sgd_resnet18_cifar10_best_id.pt
checkpoints/sgd_resnet18_cifar10_best_ood.pt
checkpoints/sam_resnet18_cifar10_last.pt
checkpoints/sam_resnet18_cifar10_best_id.pt
checkpoints/sam_resnet18_cifar10_best_ood.pt
```

## Expected Figures

```text
figures/loss_plot.png
figures/accuracy_plot.png
figures/ood_bar_chart.png
figures/per_corruption_delta.png
figures/best_vs_final_id_acc.png
figures/checkpoint_comparison.png
figures/ood_trajectory.png
figures/generalization_gap.png
figures/summary_table.png
```

`ood_trajectory.png` requires rerunning `run_experiment.py` with the updated per-epoch OOD logging. If old training logs are still present, `plot_results.py` will skip that figure and print a message.

## GitHub / Sharing Notes

Generated datasets, checkpoints, logs, metrics, and figures are ignored by Git by default. For sharing:

- upload source code and notebook files to GitHub;
- share `.pt` checkpoints through Google Drive, not GitHub;
- do not upload CIFAR-10 data, zip files, or large raw generated files.
