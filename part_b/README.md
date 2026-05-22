# Part B: Gradient-Direction Sharpness Proxy

This folder contains the Part B local sensitivity analysis for the SAM project. Following the feedback, this section does not claim that SAM is fully explained by "flatness" alone. Instead, it asks a narrower question:

> Under controlled gradient-direction perturbations, is the SAM checkpoint less sensitive than the SGD checkpoint?

This keeps Part B aligned with the overall project narrative: Part B measures local loss sensitivity, while Part C studies Hessian spectral structure and effective directions. Together, they are used to discuss which geometric properties SAM appears to change.

## Checkpoint Choice

The notebook is configured to use the Part A best-ID checkpoints:

```text
checkpoints/sgd_resnet18_cifar10_best_id.pt
checkpoints/sam_resnet18_cifar10_best_id.pt
```

This is intentional because Part B evaluates fixed in-distribution CIFAR-10 test batches. The last checkpoint is not used as the primary analysis target, since the final epoch can be worse than the best checkpoint selected in Part A. If the report discusses robustness rather than ID behavior, the same analysis can be repeated with the `best_ood` checkpoints as a sensitivity check.

## Method

Part B uses a gradient-direction perturbation sharpness proxy on fixed CIFAR-10 in-distribution batches. For each model, batch, and perturbation radius `rho`:

1. Compute the base cross-entropy loss.
2. Compute the gradient of the loss with respect to the model parameters.
3. Perturb parameters in the normalized gradient direction.
4. Compute the perturbed loss on the same batch.
5. Define the local sensitivity score as:

```text
sharpness_proxy = perturbed_loss - base_loss
```

This is a first-order, gradient-aligned perturbation proxy. It is not a full Hessian sharpness estimate, not a worst-case search over all directions, and not a global characterization of the loss landscape.

## Perturbation Radii

The sweep uses:

```text
rho = 0.001, 0.005, 0.01, 0.05
```

These values test whether the conclusion is stable from very small local perturbations up to the same order as the SAM training radius used in Part A (`SAM_RHO = 0.05`). The fixed-rho boxplot uses `rho = 0.01` as a representative local perturbation: it is large enough to show visible separation while remaining smaller than the training-time radius.

## Outputs

```text
batch_averaged_gradient_sharpness.csv
batch_level_gradient_sharpness.csv
focus_rho_summary.csv
batch_averaged_gradient_sharpness.png
batchwise_sharpness_boxplot.png
```

After rerunning the notebook with the best-ID checkpoints, these files should be refreshed and used in the final report.

## Interpretation

The expected interpretation should stay conservative:

> SAM shows lower gradient-direction local loss sensitivity than SGD across the tested `rho` values. This supports the claim that SAM changes the local geometry around the selected checkpoint, but it should be described as evidence from a sharpness proxy rather than a complete proof that SAM generalizes because it finds globally flatter minima.

This wording directly avoids the overclaim that "traditional flatness fully explains SAM." Recent work argues that SAM's generalization behavior cannot always be reduced to a single sharpness measure, so Part B should be treated as one piece of evidence and connected with Part C.

## Report Wording

Recommended wording:

> We evaluate local sensitivity using a gradient-direction perturbation sharpness proxy on fixed in-distribution CIFAR-10 batches. Lower values mean that the batch loss increases less under a controlled parameter perturbation. Across the tested perturbation radii, the SAM checkpoint exhibits lower local loss sensitivity than the SGD checkpoint, suggesting that SAM changes the local geometry of the learned solution. We do not treat this proxy as a full Hessian analysis or as a complete explanation of SAM's generalization behavior.

Avoid:

```text
Part B proves that SAM finds flatter minima.
```

Prefer:

```text
Part B provides proxy evidence that SAM is less sensitive to local gradient-aligned parameter perturbations.
```

## References

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR 2021. https://openreview.net/forum?id=6Tm1mposlrM
- Andriushchenko, M., & Flammarion, N. (2022). Towards Understanding Sharpness-Aware Minimization. ICML 2022. https://proceedings.mlr.press/v162/andriushchenko22a.html
- Kaur, S., Cohen, J., & Lipton, Z. C. (2023). On the Maximum Hessian Eigenvalue and Generalization. PMLR 187. https://proceedings.mlr.press/v187/kaur23a.html
- Wen, K., Li, Z., & Ma, T. (2023). Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization. NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/0354767c6386386be17cabe4fc59711b-Abstract-Conference.html
- Mueller, M., Vlaar, T., Rolnick, D., & Hein, M. (2023). Normalization Layers Are All That Sharpness-Aware Minimization Needs. arXiv:2306.04226. https://arxiv.org/abs/2306.04226
