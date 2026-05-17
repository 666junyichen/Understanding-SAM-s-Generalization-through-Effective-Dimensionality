# 5月17日 Part A 修改老师建议整理

## 当前实现状态

已经按“cosine scheduler + SAM rho sweep + 两张协议图”的方案完成代码更新：

```text
config.py: 新增 LR_SCHEDULER、SAM_RHO_SWEEP_VALUES、VAL_FRACTION
run_experiment.py: SGD 和 SAM 都使用 cosine scheduler，training log 新增 lr，checkpoint 保存 scheduler 设置
data.py: 新增 train/validation split loader，用于 rho sweep
run_rho_sweep.py: 新增 SAM rho sweep 脚本，输出 results/rho_sweep.csv
plot_results.py: 新增 lr_schedule_curve.png 和 rho_sweep.png
tests/test_plot_results.py: 新增两张图的测试
README.md: 更新 scheduler、rho sweep、输出文件说明
Run_PartA_Colab.ipynb: 更新 Colab 执行流程，先 sweep rho，再正式跑 Part A
```

本地已通过：

```text
python -m unittest discover -s tests -v
python -m py_compile config.py data.py run_experiment.py run_rho_sweep.py plot_results.py
python -c "import run_rho_sweep, run_experiment, plot_results; print('imports ok')"
```

还没有在本地跑完整训练，因为完整 rho sweep 和正式 SGD/SAM 训练应放到 Colab GPU 上跑。

## 最新执行决策

根据你现在的选择，建议采用这个落地版本：

```text
LR scheduler: 只使用 cosine，不同时加入 multistep
rho sweep: 新增 run_rho_sweep.py，只对 SAM 做 rho sweep
新增图: rho_sweep.png 和 lr_schedule_curve.png 都可以新增
正式实验: SGD 和 SAM 都使用同一个 cosine LR scheduler
```

具体回答你这次问的几个点：

1. 可以只在 `config.py` 新增 `LR_SCHEDULER = "cosine"`，不需要再新增 `LR_SCHEDULER = "multistep"`、`LR_MILESTONES = (30, 45)`、`LR_GAMMA = 0.1`。老师说 step decay 是标准做法之一，但 cosine 也是标准做法，而且更贴近 SAM 官方训练代码。
2. `lr` 写进 training log 不是绝对必须，但建议加。它可以证明每个 epoch 的学习率确实按 cosine 变化，也方便生成 `lr_schedule_curve.png`。
3. checkpoint 里保存 scheduler 设置也不是绝对必须，但建议加。最少保存 scheduler 名称和配置；如果想更完整，可以保存 `scheduler.state_dict()`。
4. 建议先把 cosine scheduler 的代码加好，再让 `run_rho_sweep.py` 使用同一个 cosine scheduler。这样 rho sweep 和正式实验的训练协议一致。
5. 建议一 learning-rate scheduler 要同时改 SGD 和 SAM，因为这是公平训练协议的一部分。
6. 建议二 rho sweep 只改 SAM，因为 `rho` 是 SAM 的扰动半径，普通 SGD 没有这个参数。
7. 新增 `rho_sweep.png` 和 `lr_schedule_curve.png` 是有价值的。前者证明 `rho` 不是随便选的，后者证明你回应了老师关于 fixed LR 的建议。

推荐执行顺序：

```text
1. config.py 新增 LR_SCHEDULER = "cosine"
2. run_experiment.py 给 SGD 和 SAM 都加 cosine scheduler
3. training log 新增 lr 列
4. checkpoint 保存 scheduler 配置
5. 新增 run_rho_sweep.py，只 sweep SAM rho
6. 输出 results/rho_sweep.csv
7. plot_results.py 新增 rho_sweep.png 和 lr_schedule_curve.png
8. tests/test_plot_results.py 给新增绘图函数加简单测试
9. 用 rho sweep 选最佳 SAM_RHO
10. 用最佳 rho + cosine scheduler 正式重跑 SGD 和 SAM
11. 更新 README 和 Colab notebook
```

## 0. 这次检查了哪些材料

已检查：

```text
PART_A_CONVERSATION_SUMMARY.md
Part A/PartA reference1247/PartA reference apa7 格式.txt
Part A/PartA reference1247/reference1 Sharpness Aware Minimization for Efficiently Improv.pdf
Part A/PartA reference1247/reference2  Implicit Sharpness Aware Minimization for.pdf
Part A/PartA reference1247/reference4  Boosting sharpness aware training with dynamicneighborhood.pdf
Part A/PartA reference1247/reference7 Sharpness Aware Minimization method with momentum acceleration for.pdf
README.md
config.py
run_experiment.py
tests/test_plot_results.py
```

注意：本地 `reference1` 和 `reference4` PDF 文字提取效果不好，基本提取不到正文。因此我同时参考了 SAM 官方 GitHub README 和官方训练代码，用来判断 learning-rate scheduler 和 `rho` 的常见设置。

## 1. 老师三条建议分别属于哪里

结论：三条都和 Part A 有关，但影响范围不完全一样。

| 老师建议 | Part A 是否需要改 | 是否影响 B/C |
|---|---:|---:|
| 不要固定 `50 epochs, LR=0.1`，加 learning-rate schedule | 是。属于 Part A 训练协议 | 会影响。如果重新训练 checkpoint，B/C 最好用新 checkpoint |
| `SAM_RHO=0.05` 需要 sweep 或文献支撑 | 是。属于 Part A SAM 超参数设置 | 会影响。如果最终 rho 改了，B/C 的 SAM checkpoint 也要同步 |
| README / 报告叙事要从“做了什么-得到什么”改成“假设-证据-限制” | 是。主要改 README 和报告文字 | 会影响。B/C 要接住 Part A 的研究问题 |

所以不是 B/C 同学单独的问题。前两条主要是 Part A 实验设置；第三条是 Part A README 和整体报告叙事，但 B/C 同学也要配合。

## 2. 建议一：learning-rate scheduler 选哪个

### 参考文献和代码里看到什么

`reference2` 是 domain generalization / remote sensing 方向，实验细节里写了 SGD base optimizer，remote sensing 数据集使用 `learning rate = 0.01`、`batch size = 32`、`200 epochs`；DomainBed 部分使用随机搜索学习率。它不是 CIFAR-10 ResNet18 的标准设置，不能直接作为我们 Part A 的 scheduler 依据。

`reference7` 里有和 CIFAR / ImageNet 相关的 SAM 变体实验。它提到 ImageNet 使用 `initial learning rate = 0.1` 并用 cosine scheduling；CIFAR 实验里也使用 `learning rate = 0.1`、momentum `0.9`、perturbation strength `0.01`。这说明 SAM 系列实验不会只依赖“固定 LR=0.1 到结束”这种写法。

SAM 官方代码里默认：

```text
learning_rate = 0.1
use_learning_rate_schedule = True
lr_schedule = cosine
```

并且官方注释写得很直接：CIFAR training should use the schedule。官方 README 的 CIFAR 示例也使用 `--sam_rho 0.05`。

### 我建议你用哪个 scheduler

我建议 Part A 使用：

```text
CosineAnnealingLR(T_max=EPOCHS)
```

理由：

1. SAM 官方 JAX 代码默认就是 cosine schedule。
2. `reference7` 也提到 SAM 系列大规模实验使用 cosine scheduling。
3. cosine 比 step decay 更平滑，报告里好解释。
4. 你现在只有 50 epochs，cosine 可以自然覆盖整个训练过程，不需要争论 milestone 该放在哪里。

如果你想严格对应老师原话，也可以选择：

```text
MultiStepLR(milestones=[30, 45], gamma=0.1)
```

但我的优先推荐是 cosine，因为它和 SAM 官方代码更一致。

### 需要改哪些文件

最少需要改：

```text
config.py
run_experiment.py
README.md
Run_PartA_Colab.ipynb
```

建议在 `config.py` 新增：

```python
LR_SCHEDULER = "cosine"
```

如果用 step decay，则新增：

```python
LR_SCHEDULER = "multistep"
LR_MILESTONES = (30, 45)
LR_GAMMA = 0.1
```

在 `run_experiment.py` 中：

1. 给每个 optimizer 创建 scheduler。
2. 每个 epoch 训练完成后调用 `scheduler.step()`。
3. 在 training log 里记录当前学习率，例如新增一列 `lr`。
4. checkpoint 里保存 scheduler 设置，方便报告和复现。

`tests/test_plot_results.py` 不一定需要改，除非 `plot_results.py` 新增了依赖 `lr` 列的新图。如果只是训练 log 多一列 `lr`，现有测试一般不需要改。

## 3. 建议二：`SAM_RHO=0.05` 怎么处理

### 参考文献里看到什么

SAM 官方 README 的 CIFAR 示例使用：

```text
--sam_rho 0.05
```

MosaicML 的 SAM 文档也说，按 Foret et al. 的设置，`rho=0.05` 在 interval=1 时效果不错，等价于原始论文配置。

`reference7` 使用的 perturbation strength 是：

```text
0.01
```

这说明不同 SAM 变体 / 不同实验设置会使用不同 rho，所以只写“我们选了 0.05”说服力不够。

### 原始 SAM 是不是 sweep rho？

可以这样理解：原始 SAM 论文 / 官方实现把 `rho` 当成需要调的 SAM 超参数，官方代码输出目录也会记录 `rho`，说明它是实验配置的一部分。很多复现实验和后续论文会直接沿用 `0.05` 作为常用默认值，也有论文会 grid search 或 sensitivity analysis。

老师建议的：

```text
rho in {0.01, 0.05, 0.1, 0.2}
```

是一个合理的小范围 sweep。它不一定是每篇论文都这么做，但对你的报告很有说服力。

### rho 是不是只有 SAM 用？SGD 需不需要？

`rho` 只属于 SAM。

原因是 SAM 的目标是：

```text
min_w max_{||epsilon|| <= rho} L(w + epsilon)
```

这里的 `rho` 是权重扰动 neighborhood radius。普通 SGD 没有内部的 adversarial perturbation step，所以没有 `rho`。

SGD 需要和 SAM 保持一致的是：

```text
same model
same seed / initialization
same data split
same batch size
same base learning rate
same momentum
same weight decay
same LR scheduler
same epochs
```

但 SGD 不需要 `SAM_RHO`。

### 是否必须做 rho sweep？

最好做。如果时间允许，建议新增一个小脚本：

```text
run_rho_sweep.py
```

输出：

```text
results/rho_sweep.csv
```

建议 sweep：

```text
rho = 0.01, 0.05, 0.1, 0.2
```

注意：最好用 validation set 选 rho，而不是用 CIFAR-10 test set 或 OOD test set 选。更规范的做法是从 training set 里切出 validation set。

如果时间不够，可以保留 `SAM_RHO=0.05`，但报告必须写清楚：

```text
We use rho=0.05 following the original SAM implementation and common SAM practice. We did not conduct a full rho sensitivity analysis, which remains a limitation.
```

如果做了 sweep，而且 `0.05` 最好，可以写：

```text
We selected rho=0.05 based on a validation sweep over {0.01, 0.05, 0.1, 0.2}.
```

如果 sweep 后最佳不是 `0.05`，就把 `config.py` 里的 `SAM_RHO` 改成最佳值，并用它重跑正式 Part A。

## 4. 建议三：README 和整体叙事要改什么

### 需要改代码吗？

叙事本身主要不需要改训练代码。

需要改的是：

```text
README.md
报告 Part A 文字
可能还有 Run_PartA_Colab.ipynb 的 markdown 说明
```

如果你加了 scheduler 或 rho sweep，那才需要改代码：

```text
config.py
run_experiment.py
run_rho_sweep.py
plot_results.py
tests/test_plot_results.py
```

### `tests/test_plot_results.py` 要不要改？

目前不必因为“假设-证据-限制”的叙事去改 `tests/test_plot_results.py`。

`tests/test_plot_results.py` 的作用是测试绘图函数能不能正确生成图，不负责报告叙事。如果新增新的 hypothesis-related figure，才需要加测试。

### 是否建议新增 Part A 图？

当前 Part A 已经有这些图：

```text
accuracy_plot.png
loss_plot.png
ood_bar_chart.png
per_corruption_delta.png
best_vs_final_id_acc.png
checkpoint_comparison.png
ood_trajectory.png
generalization_gap.png
summary_table.png
```

这些已经足够支持 Part A 的主要证据。最值得新增的不是“为了假设硬画图”，而是如果你做了新实验，可以新增：

```text
rho_sweep.png
lr_schedule_curve.png
```

其中：

```text
rho_sweep.png
```

用于说明 `rho` 选择不是随便选的。

```text
lr_schedule_curve.png
```

用于说明训练协议从 fixed LR 改成了 standard scheduled LR。

如果不新增这两个实验/图，现有 `test_plot_results.py` 不用动。

## 5. 建议写进 README 的研究骨架

建议在 `README.md` 开头加入：

```text
Research question:
Does SAM-SGD improve CIFAR-10 in-distribution accuracy and simple OOD robustness compared with SGD under a matched training protocol?

Hypothesis:
SAM improves generalization because it biases training toward flatter or lower-effective-dimensional regions of the loss landscape.

Part A evidence:
Part A compares SGD and SAM-SGD on CIFAR-10 ID accuracy and OOD robustness under Gaussian noise, blur, and brightness. It establishes whether SAM shows a measurable performance advantage under the same model, data, optimizer base settings, checkpoint policy, and learning-rate schedule.

Evidence strength:
Part A supports the existence of a performance difference, especially when comparing best-ID and best-OOD checkpoints. The evidence is stronger if the final experiment uses a standard LR schedule and a justified SAM rho.

Limitation:
Part A alone cannot prove that flatness or effective dimensionality causes the observed generalization difference. It only motivates the mechanism analyses in Parts B and C.
```

## 6. 给 Part B 同学的建议文字

可以发给 Part B 同学：

```text
Part B should connect its sharpness analysis back to the main hypothesis: if SAM improves generalization by biasing training toward flatter regions, then the SAM checkpoint selected from Part A should show lower gradient-based sharpness or lower sensitivity to local perturbations than the matched SGD checkpoint. Please avoid presenting the sharpness plots as isolated results. The writing should explicitly state whether the Part B evidence supports, weakly supports, or fails to support the flatness explanation.

Recommended structure:
1. State the hypothesis being tested: SAM checkpoints should appear flatter than SGD checkpoints under a first-order sharpness proxy.
2. Report the matched checkpoint setting used from Part A, preferably best-OOD or best-ID.
3. Compare SAM vs SGD using the same batches and same perturbation definition.
4. Explain evidence strength: this is a proxy for flatness, not a complete proof of the loss landscape geometry.
5. State limitation: batch-level or gradient-based sharpness can be noisy and does not prove causality.
```

## 7. 给 Part C 同学的建议文字

可以发给 Part C 同学：

```text
Part C should connect Hessian spectrum and effective dimensionality back to the main hypothesis: if SAM improves generalization by converging to lower-dimensional or less concentrated high-curvature regions, then the SAM checkpoint should show a smaller Hessian trace, fewer dominant eigenvalues, or lower effective dimensionality than the matched SGD checkpoint.

Recommended structure:
1. State the hypothesis being tested: SAM changes the curvature spectrum of the final solution.
2. Use the same checkpoint policy as Part A/B, preferably the best-OOD checkpoint if the project focuses on robustness.
3. Compare Hessian trace, top eigenvalues, spectral concentration, and effective dimensionality for SGD vs SAM.
4. Explain evidence strength: Hessian/effective-dimensionality evidence is more mechanism-oriented than Part A, but still observational.
5. State limitation: a single checkpoint comparison cannot prove that lower effective dimensionality caused better OOD performance.
```

## 8. 推荐落地方案

### 最推荐版本

1. 在 Part A 代码中加入 cosine LR scheduler。
2. 做 `rho in {0.01, 0.05, 0.1, 0.2}` validation sweep。
3. 用最佳 rho + cosine scheduler 重跑正式 Part A。
4. 重新生成 CSV、figures、checkpoints。
5. 更新 README，把 Part A/B/C 串成“Hypothesis -> Evidence -> Limitation”。
6. 通知 B/C 同学使用新的 checkpoint。

### 时间不够版本

1. 至少加入 cosine LR scheduler 并重跑 Part A。
2. 如果来不及 rho sweep，保留 `SAM_RHO=0.05`，用 Foret et al. / SAM 官方实现作为依据，并在 limitation 里说明没有做 rho sensitivity analysis。
3. README 和报告叙事必须改，因为这是低成本高收益。

## 9. 后续实际修改清单

建议之后具体修改：

```text
config.py
run_experiment.py
README.md
Run_PartA_Colab.ipynb
```

如果做 rho sweep，建议新增：

```text
run_rho_sweep.py
results/rho_sweep.csv
figures/rho_sweep.png
```

如果新增 `rho_sweep.png` 或 `lr_schedule_curve.png`，再相应更新：

```text
plot_results.py
tests/test_plot_results.py
```
