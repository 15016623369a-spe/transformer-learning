# C02 模块验收与整理记录

验收日期：2026-10-22；环境：Conda fde，Python 3.11.16、PyTorch 2.14.0、CPU。

**结论：当前 C02 Lesson01–05 的本地教学实验与文件复核通过，可以接续 C03。** 用户已按约定通过索要作业答案确认私下完成；运行记录仅证明当前代码和固定样例的行为，不是用户独立掌握、训练效果或生产可靠性的证明。

## 本次实际执行

| 检查 | 实际结果 |
|---|---|
| Lesson01 Notebook | 新内核从头执行 13 个保存的代码单元；15 项检查通过；未处理单元格错误 0 |
| Python 学习入口 | Lesson02–05 课堂 4 个，加 Lesson03–05 参考练习 3 个，共 7 个，全部执行通过 |
| 单头 Attention 原生对照 | Lesson02 课堂与作业，共 2 组输出一致 |
| 多头 Attention 原生对照 | Lesson03 课堂 H=2、作业 H=3/H=4，共 3 组输出和逐头权重一致；拆头合并往返一致 |
| Pre-Norm Block 原生对照 | Lesson04 课堂及两种 FFN 宽度、Lesson05 课堂及作业，共 5 组输出一致 |
| 额外行为 | FFN 逐位置处理、零分支保留原始残差、因果前缀隔离、全屏蔽行拒绝均通过 |
| Lesson05 问题案例 | 别名污染复现与 clone 修正、错误 Mask 复现与恢复均通过 |

共 **10 组同参数原生对照**，采用 atol=1e-6、rtol=1e-5；无 Dropout，Block 为 GELU、norm_first=True。参考 Attention bias 清零以匹配手工无 bias 投影；双方都按本课规则清零 PAD 输出。不同 API 的布尔 Mask 按各自含义转换。

## 本次补齐与保留

- Lesson02–05 课堂脚本、已有 Lesson03/04 练习的计算和注释保持不变。
- Lesson01 保留原有学习与代码单元，修正末尾过时进度并刷新本轮执行结果。
- 新增明确标记的 Lesson05 助手参考练习、知识点、完整教师代码、作业与问题案例解释、验证和交接。
- 新增 C02 模块总览、验收记录、可重复运行的全量复核脚本，以及后续课程交接；更新仓库 README。
- 历史逐课报告保留原日期，不用新结果覆盖历史证据。最新模块结果保存于 C02-verification.json。

## 两组对照的实际差异

| 配置 | 因果开：前缀 | 因果关：前缀 | PAD 屏蔽开：真实位置 | PAD 屏蔽关：真实位置 |
|---|---:|---:|---:|---:|
| 课堂 D=8/H=2 | 0 | 0.087648 | 0 | 0.142412 |
| 作业 D=12/H=3 | 0 | 0.082388 | 0 | 0.180778 |

## 复现

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/verify.py --report /tmp/c02-verification.json
```

该脚本自动用当前解释器启动临时 Jupyter 内核，再运行全部入口并进行原生对照。默认不修改学习文件；若需要另存 Notebook，可使用 `--notebook-output /tmp/lesson01.executed.ipynb`。临时探针仅核对实际解释器与检查结果，不写入保存的学习 Notebook。

现有环境需包含 torch、nbformat、nbclient、jupyter_client、ipykernel；本次均已存在，未安装或升级依赖。没有训练、下载模型或调用付费 API。

[机器可读验收结果](C02-verification.json) · [模块说明](../C02/README.md) · [下一模块交接](C02-to-C03-handoff.md)
