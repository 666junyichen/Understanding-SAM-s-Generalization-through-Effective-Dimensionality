# SAM vs SGD on CIFAR-10

This folder contains the Part A experimental code for comparing standard SGD and Sharpness-Aware Minimization (SAM) on in-distribution and out-of-distribution image classification performance.

## Goal

Part A focuses on performance evidence:

- Train the same model with SGD and SAM-SGD.
- Evaluate both models on CIFAR-10 test data.
- Evaluate robustness under simple OOD corruptions such as noise, blur, and brightness changes.
- Save checkpoints and metrics for the flatness and effective dimensionality analyses in Parts B and C.

## Project Structure

```text
project/
├── config.py
├── utils.py
├── requirements.txt
├── README.md
├── data/
├── checkpoints/
├── results/
├── figures/
└── logs/
```

Planned experiment files:

```text
data.py
models.py
sam.py
train.py
evaluate.py
plot_results.py
run_experiment.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Planned Run Command

```bash
python run_experiment.py
```

## Expected Outputs

```text
checkpoints/sgd_resnet18_cifar10.pt
checkpoints/sam_resnet18_cifar10.pt
results/metrics.csv
results/training_log_sgd.csv
results/training_log_sam.csv
figures/loss_plot.png
figures/accuracy_plot.png
figures/ood_bar_chart.png
```

Generated datasets, checkpoints, logs, metrics, and figures are ignored by Git by default.
