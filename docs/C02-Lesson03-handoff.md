# C02 Lesson03 交接

更新时间：2026-10-22（Asia/Shanghai）。

## 当前课程状态

- Lesson01、Lesson02 已完成并于此前同步至 transformer-learning。
- Lesson03 已授课；用户于 2026-10-21 索要答案，按既定约定接受为私下作业完成确认，不再要求截图。
- 用户于 2026-10-22 明确要求检查 Lesson03 文件、整理上传 GitHub，并生成标明重点的 Markdown 讲义。
- 实际文件为 `C02/lesson03.py` 与 `C02/exercise/lesson03exercise.py`。两者原有计算均运行通过，未发现本课固定输入下的核心计算错误。
- 课堂文件原样保留；练习保留原计算和学习注释，补充结果返回、主入口保护，以及明确标识为助手参考的第四题错误复现与修正。
- 用户确认、保存代码检查、助手补充和运行验证分别记录，不将参考代码或运行通过单独当作独立掌握证据。

## 文件入口

- [重点讲义](C02-Lesson03-notes.md)：完整知识点、教师代码、重点、易错点、练习与来源。
- [作业答案及检查](C02-Lesson03-homework.md)。
- [课堂脚本](../C02/lesson03.py) 与 [练习脚本](../C02/exercise/lesson03exercise.py)。
- [验证记录](C02-Lesson03-verification.json)。

## 环境与真实结果

项目 `/Users/jace/transformer-learning`；Conda fde；解释器 `/Users/jace/miniforge3/envs/fde/bin/python`。Python 3.11.16、PyTorch 2.14.0、CPU；无安装或升级依赖、训练、付费 API 调用。

两个原始文件在编辑前通过执行；最终课堂 5 项、练习两组各 6 项、第四题 4 项检查通过。H=2、H=3、H=4 三组正常 Attention 的输出与逐头权重分别与同参数原生 MultiheadAttention 对照一致。实际投影数据拆头、合并数值一致。

## 本课重点

等宽 head_dim=D/H；每头覆盖全部 Token 且投影可使用完整输入；Linear 与 reshape/transpose 的区别；[B,H,T,head_dim] 与 [B,H,T,T]；按 Key 做 Softmax；按 head_dim 缩放；Mask 的 head 广播维；合并先交换回 Token 维；输出投影；PAD Key 与 PAD Query；配置错误及形状正确但数值顺序错误。

用户追问过“两个头是否均分”：本课与标准 nn.MultiheadAttention 等宽拆分特征维；不等宽头可以自定义，但需分别实现投影、缩放、拼接，不能直接使用等宽 reshape。

## 当前接续：Lesson04

Lesson04 已在聊天正式讲授：FFN、激活函数、残差、LayerNorm、Pre-Norm 与 Post-Norm、Pre-Norm Block 教师代码；四道作业已布置。用户尚未确认 Lesson04 作业完成，本次也未检查或发布其个人 Lesson04 代码。

接下来承接 Lesson04 的问题、作业及复核。不要因 Lesson03 本次归档而倒退教学，也不要把 Lesson04 或整个 C02 自动登记完成。C02 后续实验验收通过后再进入 C03。

继续遵循：聊天中完整讲知识点、教师代码与解释、易错点，最后统一作业；索要答案视为已私下完成。用户明确要求同步后，再按对应课程范围整理、提交并核对远端。

本次同步目标为 [transformer-learning](https://github.com/15016623369a-spe/transformer-learning)，范围为 Lesson03 两个脚本、重点讲义、作业记录、验证记录、handoff 和 README。其他课程与本地文件保持原状。
