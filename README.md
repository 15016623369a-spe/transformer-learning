# Transformer Learning

Transformer 基础学习与 PyTorch 实验。课程使用 Jupyter Notebook 与 Python 脚本，沿用 Conda `fde` 环境，在 CPU 上观察模型输入与张量形状。

本仓库专门保存 Transformer 课程。此前 C01 的服务与后端实践仍保留在 [agent-learn](https://github.com/15016623369a-spe/agent-learn)，不在两个仓库之间重复搬迁。

## 课程进度

| 课程 | 状态 | 实验与记录 |
|---|---|---|
| C02 Lesson01：Token、Token ID、Embedding 与模型输入 | 已完成：用户确认作业完成；Notebook 经本次整理后从头执行验证 | [Notebook](C02/lesson01.ipynb) · [知识点](docs/C02-Lesson01-notes.md) · [作业与第四题解释](docs/C02-Lesson01-homework.md) · [交接](docs/C02-Lesson01-handoff.md) |
| C02 Lesson02：位置机制、Q/K/V、Attention 分数、缩放、Softmax、Mask | 已完成：用户确认；脚本经检查补充并运行验证 | [脚本](C02/lesson02.py) · [知识点](docs/C02-Lesson02-notes.md) · [作业答案](docs/C02-Lesson02-homework.md) · [交接](docs/C02-Lesson02-handoff.md) |

“用户确认完成”与“助手运行代码验证”分别记录，不将整理补充的代码当成用户独立实现的证据。

## 运行方式

```bash
conda activate fde
cd /Users/jace/transformer-learning
python --version
python -c "import torch; print(torch.__version__)"
```

在支持 Jupyter Notebook 的编辑器中打开 [C02/lesson01.ipynb](C02/lesson01.ipynb)，选择 `fde` 对应的 Python 解释器，再执行“重启内核并运行全部单元格”。

本机解释器：`/Users/jace/miniforge3/envs/fde/bin/python`。Notebook 中的显示名为 `fde`，但显示名不代替实际解释器选择。

本次验证使用 Python 3.11.16、PyTorch 2.14.0，全部在 CPU 上运行；没有调用付费模型或安装、升级依赖。完整运行记录见 [验证报告](docs/C02-Lesson01-verification.json)。

## Lesson01 的输入与结果

本次保留用户 Notebook 中的输入顺序：

```text
我 查询 订单
生成 订单 分析 报告
查询 退款
```

词表大小 `V=9`，批量大小 `B=3`，补齐长度 `T=4`，表示维度 `D=6`：

```text
input_ids:        [3, 4]
embedding.weight: [9, 6]
x:                [3, 4, 6]
```

Notebook 包含用户原有学习注释、扩展后的查表实践，以及本次明确标识的参考补充：不同位置的相同 Token、未知 Token、shape 练习、两类预期错误及修正。

## 目录

```text
C02/
  lesson01.ipynb
  lesson02.py
docs/
  C02-Lesson01-notes.md
  C02-Lesson01-homework.md
  C02-Lesson01-handoff.md
  C02-Lesson01-verification.json
  C02-Lesson02-notes.md
  C02-Lesson02-homework.md
  C02-Lesson02-handoff.md
  C02-Lesson02-verification.json
```

不上传密钥、`.env`、虚拟环境、Notebook 检查点和编辑器配置。后续课程继续先授课、练习与复核；用户明确要求同步时再发布。

## Lesson02 本地运行

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/lesson02.py
```

本课包含单头注意力、作业参考补充和有意触发的错误实验。正常流程通过 15 项检查，并额外与 PyTorch 原生 Attention 做两组对照；[验证记录](docs/C02-Lesson02-verification.json) 区分正常结果与预期错误。本课代码、知识点、作业答案与交接集中保存在本仓库。
