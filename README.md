# Transformer Learning

Transformer 基础学习与 PyTorch 实验仓库，使用中文讲义、Jupyter Notebook 和 Python 脚本记录从模型输入到 Pre-Norm Block 的学习过程。

**当前进度：C02 Lesson01–05 已授课并给出作业答案，2026-10-22 本地教学实验验收通过；下一步从 C03 Lesson01：真实 Tokenizer 与模型输入编码开始。C03 尚未授课或实现。**

## C02 学了什么

Token 与 Token ID → Embedding → 位置向量 → Q/K/V → 缩放点积 Attention → Padding/Causal Mask → 多头拆分与合并 → 输出投影 → FFN、残差与 LayerNorm → Pre-Norm Block → 对照实验。

实验使用小词表、随机初始化权重和 CPU，重点是形状与计算规则。当前没有训练语言模型、词表预测头或真实文本生成。

## 课程入口

| 小课 | 主题 | 状态 | 文件 |
|---|---|---|---|
| Lesson01 | Token、Token ID、Embedding 与输入 | 已授课、答案已给出，本轮复核通过 | [代码](C02/lesson01.ipynb) · [讲义](docs/C02-Lesson01-notes.md) · [答案](docs/C02-Lesson01-homework.md) |
| Lesson02 | 位置、Q/K/V、Attention、Softmax 与 Mask | 已授课、答案已给出，本轮复核通过 | [代码](C02/lesson02.py) · [讲义](docs/C02-Lesson02-notes.md) · [答案](docs/C02-Lesson02-homework.md) |
| Lesson03 | 多头注意力与输出投影 | 已授课、答案已给出，本轮复核通过 | [代码](C02/lesson03.py) · [讲义](docs/C02-Lesson03-notes.md) · [答案](docs/C02-Lesson03-homework.md) |
| Lesson04 | FFN、残差、LayerNorm 与 Block | 已授课、答案已给出，本轮复核通过 | [代码](C02/lesson04.py) · [讲义](docs/C02-Lesson04-notes.md) · [答案](docs/C02-Lesson04-homework.md) |
| Lesson05 | 综合实验与验收 | 已授课、答案已给出，本轮复核通过 | [代码](C02/lesson05.py) · [讲义](docs/C02-Lesson05-notes.md) · [答案](docs/C02-Lesson05-homework.md) |

- [C02 模块说明与重点](C02/README.md)
- [C02 验收与整理记录](docs/C02-acceptance.md) · [机器可读结果](docs/C02-verification.json)
- [后续课程交接：C02 → C03 与完整主线](docs/C02-to-C03-handoff.md)
- [Lesson05 参考练习](C02/exercise/lesson05exercise.py)

## 运行与复现

沿用已有 Conda fde 环境。本轮实际版本：Python 3.11.16、PyTorch 2.14.0、nbformat 5.11.1、nbclient 0.11.0。

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/lesson05.py
python C02/exercise/lesson05exercise.py
python C02/verify.py --report /tmp/c02-verification.json
```

本机解释器为 `/Users/jace/miniforge3/envs/fde/bin/python`。Notebook 在编辑器中选择同一个解释器，再重启内核并运行全部；显示名 fde 不代替解释器核对。

全量复核会以当前解释器启动临时 Jupyter 内核，执行 Lesson01 的 13 个代码单元和 15 项检查，再运行 7 个 Python 学习入口并完成 10 组同参数 PyTorch 原生对照。默认不改学习文件，不下载模型，不训练，不调用付费 API。

## 文件组织

```text
C02/
  README.md                 # 模块知识图谱、约定与运行入口
  lesson01.ipynb
  lesson02.py
  lesson03.py
  lesson04.py
  lesson05.py
  verify.py                 # Notebook + 脚本 + 原生实现全量复核
  exercise/
    lesson03exercise.py
    lesson04exercise.py
    lesson05exercise.py
docs/
  C02-Lesson01-...           # notes / homework / handoff / verification.json
  C02-Lesson02-...
  C02-Lesson03-...
  C02-Lesson04-...
  C02-Lesson05-...
  C02-acceptance.md
  C02-verification.json
  C02-to-C03-handoff.md
```

## 记录约定与项目关系

用户索要答案按授课约定接受为私下完成确认；原有文件检查、助手参考补充、运行验证分别记录，不把参考代码通过当作独立掌握证据。旧逐课交接保留各自日期的历史状态，最新接续见模块交接。

本仓库用于 Transformer 学习；C01 服务与后端实践仍在 [agent-learn](https://github.com/15016623369a-spe/agent-learn)。主项目为“企业知识驱动的办公分析与报告 Agent”，后续会学习真实推理、工具、SQL、RAG、Agent、持久化、审批恢复与部署。

后续课程先在聊天完整讲知识点、教师代码、解释和易错点，最后统一布置作业；用户明确要求同步后再发布。密钥、.env、环境、缓存与编辑器配置由 .gitignore 排除。
