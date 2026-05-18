# 5月17日 Part A 修改老师建议整理

## 当前实现状态

这版采用较轻量、可完成的修改方案：

```text
保留: SGD 和 SAM 都使用 cosine learning-rate scheduler
保留: training log 记录 lr
保留: checkpoint 记录 scheduler 设置
保留: plot_results.py 生成 lr_schedule_curve.png
保留: SAM_RHO = 0.05
删除: SAM rho 搜索脚本、CSV 输出和对应图片
```

这样做的原因是：完整尝试多个 `rho` 值会让训练成本变成多次 SAM 训练，时间开销太大。为了按时完成 Part A，我们固定使用 `SAM_RHO = 0.05`，并在报告中说明它沿用原始 SAM / 常见 CIFAR 风格实验设置。

## 老师三条建议对应的处理方式

| 老师建议 | 当前处理 |
|---|---|
| 不要固定 `50 epochs, LR=0.1` | 已改。SGD 和 SAM 都使用 cosine scheduler |
| `SAM_RHO=0.05` 需要更有说服力 | 不做额外搜索。报告中说明采用原始 SAM 常用设置，并把未做敏感性分析写成 limitation |
| README / 报告叙事要有研究骨架 | README 和报告应按“假设 -> 证据 -> 限制”组织 |

## 需要在报告里解释 `rho=0.05`

建议写法：

```text
We set the SAM neighborhood radius to rho = 0.05, following the original SAM implementation and common CIFAR-style SAM practice. Due to computational constraints, we did not perform a full rho sensitivity analysis. Therefore, our results should be interpreted as evaluating SAM under a standard default rho rather than as evidence that rho = 0.05 is optimal for this setting.
```

中文理解：

```text
我们不是说 0.05 一定最优，而是说 0.05 是 SAM 常用默认设置。
因为时间和计算资源限制，没有额外做超参数敏感性分析。
这是一个合理设置，但也是报告里的 limitation。
```

## Part A 现在需要生成的新增图

只保留：

```text
figures/lr_schedule_curve.png
```

这张图用来证明训练不再是固定 `LR=0.1`，而是随 epoch 使用 cosine schedule 下降。

不再生成额外的 rho 搜索结果表或 rho 搜索图片。

## Part A/B/C 叙事骨架

```text
Hypothesis:
SAM improves generalization because it biases training toward flatter or lower-effective-dimensional regions of the loss landscape.

Part A evidence:
Compare SGD and SAM on CIFAR-10 ID accuracy and OOD robustness. This establishes whether SAM actually shows a performance advantage under our setting.

Part B evidence:
Analyze gradient-based sharpness / local perturbation behavior. This checks whether SAM checkpoints appear flatter under a first-order sharpness proxy.

Part C evidence:
Analyze Hessian spectrum and effective dimensionality. This checks whether SAM's solution concentrates curvature differently and whether this aligns with the effective-dimensionality hypothesis.

Conclusion boundary:
The experiments can support an association between SAM, robustness, and flatter or lower-dimensional geometry, but they cannot prove a causal mechanism by themselves.
```

## 当前需要修改的文件

```text
config.py
data.py
run_experiment.py
plot_results.py
tests/test_plot_results.py
README.md
Run_PartA_Colab.ipynb
5月17日partA修改老师建议.md
```
