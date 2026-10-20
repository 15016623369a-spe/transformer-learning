# C02 Lesson02：位置机制与单头 Self-Attention

课堂代码见 [lesson02.py](../C02/lesson02.py)，[作业答案](C02-Lesson02-homework.md) 单独保存。本课只做前向计算与观察，参数为随机初始化，没有训练。

## 1. Token 向量与位置向量

同一个 Token ID 查同一张 Embedding 表时，得到同一个 Token 向量。为了让后续计算可以使用位置，本课把可学习的绝对位置向量加到 Token 向量上：

```python
position_ids = torch.arange(T, device="cpu")
token_x = token_embedding(input_ids)        # [B,T,D]
position_x = position_embedding(position_ids)  # [T,D]
h = token_x + position_x.unsqueeze(0)       # [B,T,D]
```

`unsqueeze(0)` 添加批次维，所有输入共享同一组位置编号。它不会把原来的 Token ID 改成另一个 ID。位置表的容量必须覆盖本次位置编号。

`padding_idx=0` 只约束 Token Embedding 的 PAD 行；加上位置向量后，PAD 位置的 `h` 通常不再为零，因此仍需要显式 Mask。

## 2. 同一份输入得到不同的 Q、K、V

```python
query = q_proj(h)  # [B,T,d_k]
key = k_proj(h)    # [B,T,d_k]
value = v_proj(h)  # [B,T,d_v]
```

Self-Attention 的三者来自同一份 `h`，但使用不同线性层。Query 与 Key 计算匹配分数，Value 提供被汇总的信息。Q 和 K 做点积，特征维度必须相同；Value 的维度可以不同。

本课教师例子为 `B=2, T=4, D=4, d_k=2, d_v=3`。这些线性层使用 `bias=False`，便于观察。

## 3. 分数、缩放、归一化、加权汇总

```python
scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
context = weights @ value
```

| 对象 | shape | 含义 |
|---|---|---|
| Q | [B,T,d_k] | 每个位置的 Query |
| K 转置后 | [B,d_k,T] | 供每个 Query 与各 Key 做点积 |
| scores | [B,T,T] | 行为 Query 位置，列为 Key 位置 |
| weights | [B,T,T] | 每个 Query 对允许的 Key 分配的权重 |
| context | [B,T,d_v] | 按权重汇总各个 Value |

`scores[b,i,j]` 表示输入 b 的 Query i 与 Key j 的缩放匹配分数。`transpose(-2,-1)` 只交换最后两个维度，保留批次维。

除以 `sqrt(d_k)` 可以缓和点积随维度增大而扩大的尺度，减少 Softmax 过早集中。Softmax 沿最后一维，即 Key 位置，对每个 Query 单独归一化。Attention 权重不是训练后的正确性保证；本课随机参数算出的权重也没有经过语义学习。

## 4. Padding Mask 与 Causal Mask

本课统一约定：`allowed=True` 表示允许读取，`masked_fill` 的条件为 True 时则执行替换，因此使用 `~allowed`。

```python
valid_mask = input_ids.ne(0)          # [B,T]
key_allowed = valid_mask.unsqueeze(1)  # [B,1,T]
causal_allowed = torch.ones(T, T, dtype=torch.bool).tril()
allowed = key_allowed & causal_allowed.unsqueeze(0)  # [B,T,T]
```

Padding Mask 禁止读取 PAD Key；Causal Mask 保留主对角线及其下方，使第 i 个 Query 只能读到位置 j≤i。二者取交集。只加 Padding Mask 时，真实 Token 仍能读取未来的真实 Token。

禁止位置在 Softmax 之前填入负无穷；其指数为零。填 0 仍会分到权重，因为 exp(0)=1。

屏蔽 PAD Key 不会自动清空 PAD Query 的输出。本课在汇总后另行清零：

```python
output = context.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
```

## 5. 全屏蔽行与检查方法

如果某个 Query 的所有 Key 都被禁止，直接对全是负无穷的一行手工 Softmax 会产生 NaN。脚本的 `masked_attention()` 在 Softmax 前检查每行至少存在一个允许位置，不满足就抛出清楚的 `ValueError`。教师例子采用右侧补齐，并要求每条输入至少有一个真实 Token。

正常流程检查 shape、权重逐行和、禁止位置权重为零、输出有限，以及 PAD Query 的最终输出为零。只检查行和为 1 不足以证明 Mask 正确。

易错点还包括：沿 `dim=-2` 在不同 Query 之间归一化；把分数当作概率；遗漏缩放；把行列含义写反；假设 PAD 向量加位置后仍为零。不同 PyTorch 接口的布尔 Mask 约定可能不同，迁移代码时需要核对具体接口。

## 6. 运行与代码导航

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/lesson02.py
```

脚本前半部分保留课堂示例，后半部分明确标记为助手整理的参考补充：`run_manual_exercise()` 核对手算，`run_homework_attention()` 完成指定输入实践，`run_error_experiments()` 展示三类预期错误，主入口汇总 15 项检查。

正常流程通过 15 项检查；教师例子与作业例子另与 PyTorch 原生 Attention 做两组对照，见 [验证记录](C02-Lesson02-verification.json)。错误实验中的 NaN 为有意复现，不能当作正常输出。完整题目答案和修正原因见 [作业记录](C02-Lesson02-homework.md)。

下一小课接多头注意力，尚未授课。
