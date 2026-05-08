# Part B: Flatness Analysis

This folder contains the flatness analysis for the SAM project. It uses the trained SGD and SAM-SGD checkpoints produced by Part A and compares their local loss sensitivity.

## Method

Part B uses a gradient-direction perturbation sharpness proxy on fixed CIFAR-10 in-distribution test batches.

For each model, batch, and perturbation radius `rho`:

1. Compute the base loss.
2. Compute the gradient of the loss with respect to model parameters.
3. Perturb parameters in the normalized gradient direction.
4. Compute the perturbed loss.
5. Define sharpness as:

```text
sharpness = perturbed_loss - base_loss
```

This is not a full Hessian sharpness estimate and not a worst-case sharpness over all directions. It is a controlled perturbation-based proxy for local flatness.

## Inputs

The notebook expects the Part A checkpoints:

```text
checkpoints/sgd_resnet18_cifar10.pt
checkpoints/sam_resnet18_cifar10.pt
```

The checkpoints are not committed to GitHub because they are large model artifacts. Share them through Google Drive.

## Outputs

```text
batch_averaged_gradient_sharpness.csv
batch_level_gradient_sharpness.csv
focus_rho_summary.csv
batch_averaged_gradient_sharpness.png
batchwise_sharpness_boxplot.png
```

## Key Finding

SAM-SGD shows lower gradient-direction sharpness than SGD across the tested perturbation radii. For example, at `rho = 0.01`:

```text
SGD mean sharpness = 0.0933
SAM mean sharpness = 0.0382
```

This supports the traditional explanation that SAM tends to find flatter minima.

## Report Wording

Recommended wording:

> We evaluate flatness using a gradient-direction perturbation-based sharpness proxy on fixed in-distribution batches. Lower sharpness indicates that the loss increases less under a controlled local perturbation.

Avoid claiming that this is a full Hessian analysis or an exact global characterization of the loss landscape.
