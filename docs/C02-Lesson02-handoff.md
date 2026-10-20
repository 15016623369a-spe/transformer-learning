# C02 Lesson02 交接

更新时间：2026-10-20（Asia/Shanghai）。

## 当前状态与依据

- C02 Lesson01 已完成并在此前同步 GitHub。
- C02 Lesson02 已授课；用户确认作业完成，随后要求答案、检查代码并补充缺少部分。
- 本次实际文件为 `C02/lesson02.py`。原有教师示例运行正确，保留原代码及注释。
- 本次参考补充：第二题手算核对、第三题指定输入与维度、第四题三个错误实验与保护、15 项运行检查。
- 用户确认、原代码检查、助手补充、运行证据分别记录；不将助手补充代码当成用户独立实现的证据。
- 用户于 2026-10-20 明确要求同步 GitHub；本次发布范围为 Lesson02 脚本、知识点讲义、作业答案、验证记录、交接与 README。

## 环境与验证

- 本地项目：`/Users/jace/transformer-learning`。
- Conda fde，解释器 `/Users/jace/miniforge3/envs/fde/bin/python`。
- Python 3.11.16，PyTorch 2.14.0，CPU；无依赖升级、训练或付费 API 调用。
- 正常计算的 shape、权重归一化、禁止位置为零、PAD Query 输出为零、有限值均已检查。
- 两组正常 Attention 与 PyTorch 原生实现对照。
- 错误实验中的 NaN 是刻意展示的错误现象；全 PAD 输入被保护逻辑明确拒绝。
- 课堂知识与代码说明见 [讲义](C02-Lesson02-notes.md)。
- 真实结果见 [验证记录](C02-Lesson02-verification.json)，答案见 [作业记录](C02-Lesson02-homework.md)。

## 本课知识

可学习绝对位置向量；同源不同投影的 Q/K/V；[B,T,d_k] 与转置 Key 相乘得到 [B,T,T]；sqrt(d_k) 缩放；沿 Key 位置的 Softmax；Value 加权汇总；Padding／Causal Mask；Mask 的行列与布尔约定；PAD Key 与 PAD Query 的区别；全屏蔽行保护。

特别注意：Softmax 行和为 1 不足以证明 Mask 正确；换错 Softmax 轴有时数值有限仍然错误；右补齐的有效 Query 在因果限制下可能已经看不到 PAD，不能据此断定 PAD Mask 可删。

## 下一课

C02 下一小课接多头注意力：head 维度、拆分与合并、投影、shape 跟踪。尚未授课或验收，不自动进入 C03。

继续先完整讲知识点，再给教师代码与解释、易错点，最后统一作业。按用户确认接受私下作业完成，不重复要求截图。

同步仓库为 [transformer-learning](https://github.com/15016623369a-spe/transformer-learning)。本次同步依据用户的明确请求；后续课程仍需用户明确要求同步时再发布，并核对远端结果、保留其他本地改动。
