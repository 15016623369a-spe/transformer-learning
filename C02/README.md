# C02：Transformer 基础

更新日期：2026-10-22。Lesson01–05 的当前本地教学实验验收通过，后续从 C03 Lesson01 开始。

## 本模块目标

能够沿计算链说明数据如何从 Token ID 变成上下文表示，解释各张量维度，以及通过对照实验检查信息访问规则。本模块不训练语言模型，不包含词表预测头、真实 Tokenizer 或生成循环。

## 五课索引

| 小课 | 知识点 | 实践 | 资料 |
|---|---|---|---|
| Lesson01 | Token、Token ID、Embedding 与输入 | Notebook 查表、PAD/UNK、输入类型与编号范围 | [讲义](../docs/C02-Lesson01-notes.md) · [答案](../docs/C02-Lesson01-homework.md) |
| Lesson02 | 位置、Q/K/V、Attention、Softmax 与 Mask | 单头计算、手算、Mask 错误与原生对照 | [讲义](../docs/C02-Lesson02-notes.md) · [答案](../docs/C02-Lesson02-homework.md) |
| Lesson03 | 多头注意力与输出投影 | 等宽拆头、正确合并、H=3/4 对照 | [讲义](../docs/C02-Lesson03-notes.md) · [答案](../docs/C02-Lesson03-homework.md) |
| Lesson04 | FFN、残差、LayerNorm 与 Block | Pre-Norm、两种 FFN 宽度、维度与归一化错误 | [讲义](../docs/C02-Lesson04-notes.md) · [答案](../docs/C02-Lesson04-homework.md) |
| Lesson05 | 综合实验与验收 | 固定参数、因果与 PAD 对照、clone 与全屏蔽行 | [讲义](../docs/C02-Lesson05-notes.md) · [答案](../docs/C02-Lesson05-homework.md) |

## 数据流与形状

| 阶段 | 形状 | 解释 |
|---|---|---|
| Token ID | [B,T] | 整数编号，不是语义强弱 |
| Embedding 表 | [V,D] | V 个 Token 对应的 D 维参数 |
| Token + 位置表示 | [B,T,D] | 加入位置后区分相同 Token 的不同位置 |
| Q/K/V 投影 | [B,T,D] | 多头课堂示例的总投影宽度 |
| 拆头 | [B,H,T,D/H] | 等宽划分特征，每头仍覆盖全部位置 |
| 分数与权重 | [B,H,T,T] | 最后两维为 Query 位置、Key 位置 |
| 允许矩阵 | [B,1,T,T] | head 维广播，各头共享访问规则 |
| 合并及输出投影 | [B,T,D] | 先恢复位置在前，再拼接各头特征 |
| FFN 中间 | [B,T,D_ff] | 逐位置的非线性变换 |
| Block 输出 | [B,T,D] | 两次残差后保持主干宽度 |

B=批量大小、T=补齐后的长度、V=词表大小、D=表示宽度、H=头数。本课标准等宽写法要求 D 能被 H 整除；自定义不等宽头需另写投影、缩放与合并。

Lesson02 单头独立示例允许 d_k、d_v 与 D 不同，不能把上表的多头配置强套到每一个实验。Lesson01 保存的输入为“我 查询 订单”“生成 订单 分析 报告”“查询 退款”，对应 B=3/T=4/V=9/D=6。

## 必须区分的概念

- 空格 split 是教学分词；真实子词 Tokenizer 在 C03 学习。
- UNK 表示未收录内容，仍是有效 Token；PAD 用于补齐。
- 随机 Token Embedding 不是训练后语义表示，也不直接等同于 RAG 检索向量。
- Linear 生成可学习特征；reshape/transpose 改变维度组织。
- Attention 汇集允许位置的信息；FFN 不跨位置混合，参数在各位置共享。
- LayerNorm(D) 对每个 Token 的 D 个特征独立归一化，方差对应 correction=0，不产生概率。

## 计算公式

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
\]

M 在允许位置为 0，在禁止位置为负无穷；Softmax 沿最后的 Key 维计算。多头时 d_k 为每个头的点积宽度。

```python
# Pre-Norm；实际代码还在两次残差后清零 PAD。
u = x + attention(norm1(x))
y = u + ff_down(GELU(ff_up(norm2(u))))
```

## Mask 与 PAD 约定

1. 手工代码的 allowed=True 表示允许；masked_fill 使用 ~allowed。
2. functional scaled_dot_product_attention 的布尔 Mask 同样 True=允许；本次原生 MultiheadAttention/TransformerEncoderLayer 对照转换为 True=禁止。不能把不同 API 的布尔含义混用。
3. 采用非空输入、右侧补齐；每个 Query 至少有一个可读 Key。全屏蔽行在 Softmax 之前拒绝。
4. 屏蔽 PAD Key 防止被读取；清零 PAD 输出是另一个动作。FFN bias 和 LayerNorm 仿射参数可能让内部 PAD 表示非零。
5. 因果实验仅改未来位置、比较前缀；PAD 实验在初始清零后扰动，并暂时关闭因果限制以单独检查 PAD Key Mask。

## 运行入口

在仓库根目录运行：

```bash
conda activate fde
python C02/lesson02.py
python C02/lesson03.py
python C02/exercise/lesson03exercise.py
python C02/lesson04.py
python C02/exercise/lesson04exercise.py
python C02/lesson05.py
python C02/exercise/lesson05exercise.py
python C02/verify.py --report /tmp/c02-verification.json
```

Lesson01 使用 Notebook；全量复核脚本会启动新的内核从头执行它。依赖沿用现有 fde，不另建环境、不自动升级。

## 当前验收范围

13 个保存的 Notebook 代码单元、15 项 Notebook 检查、7 个 Python 学习入口、10 组同参数原生实现对照均通过，另包含因果隔离、FFN 逐位置、残差旁路、全屏蔽行保护和问题案例修正。

这说明当前固定样例与计算规则符合预期，不代表用户独立实现证明、训练效果或任意输入下的生产可靠性。课堂文件与助手参考补充在文件和文档中区分。

[验收说明](../docs/C02-acceptance.md) · [完整结果](../docs/C02-verification.json) · [C03 与后续课程交接](../docs/C02-to-C03-handoff.md)
