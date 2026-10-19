# C02 Lesson01 交接

更新时间：2026-10-19（Asia/Shanghai）。

## 当前状态

- C02 Lesson01：已完成。用户确认完成作业，并在对照答案及第四题解释后再次确认完成。
- Notebook：保留用户原有注释与三条输入顺序；本次助手修正 PAD 检查、更新执行输出，并补充明确标识的参考单元格。
- 代码验证：实际执行结果与环境记录见 [验证报告](C02-Lesson01-verification.json)。
- 上述用户确认、原代码检查、助手补充及运行证据分别记录，不将参考答案代码的存在当成独立掌握证明。
- C02 Lesson02：待开始。

## 路径与发布目标

- 本地项目：`/Users/jace/transformer-learning`。
- Notebook：`C02/lesson01.ipynb`。
- 本次用户明确指定的仓库：[transformer-learning](https://github.com/15016623369a-spe/transformer-learning)。
- 此次指定取代旧交接中“C02 仍放 agent-learn”的目录与发布安排；C01 原项目及历史继续保留，未迁移。
- 课程实验不追加到旧 FastAPI 的 main.py。

## 环境

- Conda：fde。
- 本机解释器：`/Users/jace/miniforge3/envs/fde/bin/python`。
- Python：3.11.16；PyTorch：2.14.0。
- CPU 实验，未调用付费模型，未安装或升级依赖。

## 本次发现与修正

1. 用户保存后，原 Notebook 共 17 个单元格；已改为九项词表、三条输入、D=6，但保留了旧的两条输入输出。
2. 原来的 `x[1,2]` 曾是 PAD；用户第二条输入现在长度为四，该位置已对应“分析”。改为 `x[~valid_mask]`，以实际标记提取全部 PAD。
3. 保留当前输入顺序：我 查询 订单；生成 订单 分析 报告；查询 退款。
4. 助手参考补充：不同位置的相同 Token、UNK 与 PAD、第二题 shape 核对、第四题两类预期错误及修正、最终运行检查。
5. 从头执行 Notebook，避免将旧内核变量或旧输出作为当前结果。

## 已讲内容

Token／ID／Embedding；B、T、V、D；[B,T]→[B,T,D]；词表映射与 dtype；PAD／UNK；随机初始化与训练；查表向量与上下文表示；Token 向量与常见 RAG 文本检索向量的区别。

## 下一课

从 **C02 Lesson02：位置机制、Q/K/V、Attention 分数、缩放、Softmax、Mask** 开始。按理解情况拆分；后续再学多头注意力、FFN、残差与归一化。

教学顺序继续为：聊天中完整讲知识点 → 教师代码与解释 → 易错点 → 最后统一作业。用户索要答案表示已私下完成，接受确认，不重复要求截图。

不自动推进 C03，不把尚未进行的课程登记完成。先授课与练习；用户明确要求同步时，再整理当前课程范围并上传。
