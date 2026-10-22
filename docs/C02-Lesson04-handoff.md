# C02 Lesson04 交接

更新时间：2026-10-22（Asia/Shanghai）。

## 当前状态

- Lesson01–03 已完成并同步 transformer-learning；Lesson04 已在聊天讲授并给出四题答案。
- 用户索要答案按既定约定接受为私下完成确认；本次明确要求检查个人文件并同步 GitHub。
- 实际保存的 Lesson04 文件为 C02/lesson04.py，课堂配置 D=8、H=2、D_ff=16。原有 7 项检查通过，核心计算正确，保留原文件。
- 没有发现独立的 Lesson04 作业文件；C02/exercise/lesson04exercise.py 是本次新增的助手参考补充，含手算、D_ff=24/48 两组和两个错误案例。不能将其当作用户原有独立实现。
- 知识点、完整教师代码、重点与易错点在 notes；四题答案和检查结论在 homework；实际运行结果在 verification.json。

## 文件入口

- [课堂脚本](../C02/lesson04.py)
- [作业参考补充](../C02/exercise/lesson04exercise.py)
- [重点讲义](C02-Lesson04-notes.md)
- [作业答案与文件检查](C02-Lesson04-homework.md)
- [验证记录](C02-Lesson04-verification.json)

## 环境与验证

项目 /Users/jace/transformer-learning；Conda fde；解释器 /Users/jace/miniforge3/envs/fde/bin/python。Python 3.11.16、PyTorch 2.14.0、CPU。

课堂与两种 FFN 宽度均通过本例检查；三组输出与同权重、dropout=0、GELU、norm_first=True 的原生 TransformerEncoderLayer 对照一致，按本课规则清零 PAD。LayerNorm 手算、未来位置隔离、零分支残差、FFN 逐位置处理、全屏蔽行拒绝及错误修正通过。无依赖变更、训练或付费 API。

## 本课重点

FFN 的 D→D_ff→D；激活函数的非线性；原输入旁路与两次残差；LayerNorm 的末尾 D 维统计、correction=0、eps、gamma/beta；Pre-Norm/Post-Norm 顺序；Mask 语义转换；PAD Key 屏蔽与两次残差后清零。

## 后续接续

Lesson04 的授课、答案和当前文件复核已完成，下一步承接 C02 的实验与验收；若用户提出本课问题，先解答。不要自动宣布整个 C02 已完成，也不要未经 C02 验收直接跳到 C03。私下作业确认、原文件检查、助手补充与实际运行验证分别记录。

继续遵循：聊天中先完整讲知识点、教师代码、解释和易错点，最后统一布置作业；索要答案视为私下完成，不要求截图证明。后续用户要求同步时，再按对应课程范围发布。

本次同步目标为 https://github.com/15016623369a-spe/transformer-learning，范围为 Lesson04 课堂文件、参考练习、重点讲义、作业答案、验证记录、handoff 和 README。
