# C02 Lesson03：多头注意力（重点讲义）

整理日期：2026-10-22（Asia/Shanghai）。配套文件：[课堂脚本](../C02/lesson03.py)、[练习脚本](../C02/exercise/lesson03exercise.py)、[作业答案](C02-Lesson03-homework.md)、[验证记录](C02-Lesson03-verification.json)。

## 本课必须掌握的五个重点

1. **每个头仍然包含全部 Token 位置。** 常见写法分配的是特征维度：`head_dim = D // H`。
2. **先用可学习投影生成 Q/K/V，再拆头。** 每个头的投影都能使用完整 D 维输入；等宽拆分不等于让头只能读取原输入的一小段。
3. **拆头与合并要保持数值对应。** `[B,T,D] → [B,T,H,head_dim] → [B,H,T,head_dim]`；合并时先交换回 Token 位置在前的顺序。
4. **每个头沿 Key 位置做 Softmax，用 `sqrt(head_dim)` 缩放。** 分数与权重为 `[B,H,T,T]`，最后两维依次是 Query、Key。
5. **Mask 为 head 维留广播位置。** 本课 `allowed` 为 `[B,1,T,T]`，True 表示允许；屏蔽 PAD Key 与清零 PAD Query 输出是两个步骤。

## 1. 从单头到多头

单头为每条输入计算一张 `[T,T]` 权重矩阵；多头让每条输入拥有 H 张这样的矩阵。各头通过不同投影计算自己的分数与权重，再将输出拼接并投影。

```text
同一份 h [B,T,D]
    ↓ 三个独立的 Q/K/V 投影
Q_all、K_all、V_all [B,T,D]
    ↓ 拆头
Q、K、Value [B,H,T,head_dim]
    ↓ 各头 Attention
head_output [B,H,T,head_dim]
    ↓ 交换维度、拼接
merged [B,T,D]
    ↓ 输出投影
output [B,T,D]
```

不同头可以形成不同的信息组合。不能预先断言某个头固定负责语法或某类业务关系；本课参数随机初始化，也没有经过语义训练。

## 2. D、H、head_dim 与“平均分”

| 符号 | 含义 | 教师例子 |
|---|---|---:|
| B | 批量大小 | 2 |
| T | 补齐后的 Token 长度 | 4 |
| D | 总表示维度 | 8 |
| H | 头数 | 2 |
| head_dim | 每个头的特征维度 | 4 |

**重点：本课及标准 `nn.MultiheadAttention` 使用等宽拆分，要求 D 能被 H 整除。** D=8、H=2 得到每头 4 维，均分的是维度数量，不会把向量的数值除以 2，也不会把句子切成两段。

原理上可以自行实现不同宽度的头，例如 3 维和 5 维，但需要分别处理各头的投影、Q/K 点积维度、缩放和输出拼接；不能直接套用本课的等宽 reshape。每个头内部的 Q 与 K 必须具有相同的点积维度。

## 3. 为什么一个 Linear 可以一次生成所有头

```python
q_proj = nn.Linear(D, D, bias=False)
q_all = q_proj(h)  # [B,T,D]
```

一个大的线性层可以按输出特征分组，容纳不同头的投影参数。以 D=8、H=2 为例，投影输出的前 4 维供 head 0 使用，后 4 维供 head 1 使用。每个输出特征都可以使用完整输入的 8 个特征进行计算。

**重点：Linear 生成特征，有可学习参数；reshape 与 transpose 组织维度，没有可学习参数。**

## 4. 拆头与矩阵乘法

```python
query = q_all.reshape(B, T, H, head_dim).transpose(1, 2)
scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
```

形状依次为：

```text
[B,T,D] → [B,T,H,head_dim] → [B,H,T,head_dim]
[B,H,T,head_dim] @ [B,H,head_dim,T] → [B,H,T,T]
```

`scores[b,a,i,j]` 表示第 b 条输入、第 a 个头的 Query i 与 Key j 的缩放分数。B 与 H 的各组分别参与矩阵乘法，没有在这里跨批次或跨头混合。

```python
weights = torch.softmax(masked_scores, dim=-1)
head_output = weights @ value
```

**重点：Softmax 的最后一维是 Key 位置。** 每个固定的批次、头、Query 都单独分配一行权重。本课没有 Dropout，正常权重逐行和约为 1。

## 5. Mask 与 PAD

```python
key_allowed = valid_mask[:, None, None, :]  # [B,1,1,T]
causal = torch.ones(T, T, dtype=torch.bool).tril()
allowed = key_allowed & causal[None, None, :, :]  # [B,1,T,T]
```

`allowed` 的 head 维为 1，可以共享给 H 个头；各头仍然根据自己的分数得到不同权重。因果限制保留主对角线，禁止读取未来位置。

```python
masked_scores = scores.masked_fill(~allowed, float("-inf"))
```

**重点：本课 True=允许，执行替换时用 `~allowed`。** 禁止位置在 Softmax 之前填负无穷，不能填 0。每行必须至少有一个允许的 Key，否则手工 Softmax 会产生 NaN；本例采用右补齐，且每条输入至少有一个真实 Token。

屏蔽 PAD Key 后，PAD Query 仍可读取前面的真实 Key。本课在输出投影后另行清零 PAD Query：

```python
output = projected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
```

## 6. 合并与输出投影

```python
merged = head_output.transpose(1, 2).contiguous().view(B, T, D)
projected = out_proj(merged)
```

先从 `[B,H,T,head_dim]` 恢复为 `[B,T,H,head_dim]`，再将同一个 Token 的 H 个头拼接起来。输出投影进一步组合各头特征，输出仍是 `[B,T,D]`。

**重点：形状正确不足以说明合并正确。** 直接对 `[B,H,T,head_dim]` reshape 成 `[B,T,D]`，可能将不同 Token 的特征排到一起。必须同时检查数值对应关系。

转置后的存储布局可能不支持直接 view，故本课先使用 contiguous；也可以在正确转置后用 reshape。

## 7. 完整教师代码

以下保留本次检查时的课堂脚本内容。代码只进行 CPU 前向计算，不训练，不调用付费 API。

```python
import math
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)

# 1. 沿用前两课的词表与输入。
vocab = [
    "<PAD>", "<UNK>", "我", "查询", "订单",
    "生成", "报告", "退款", "分析",
]
input_ids = torch.tensor(
    [[2, 3, 4, 0], [5, 4, 8, 6]],
    dtype=torch.long,
    device="cpu",
)
valid_mask = input_ids.ne(0)
B, T = input_ids.shape

D, H = 8, 2
if H <= 0 or D % H != 0:
    raise ValueError("本课要求 H 为正整数，且 D 能被 H 整除")
head_dim = D // H

# 2. Token 向量 + 位置向量。
token_embedding = nn.Embedding(
    len(vocab), D, padding_idx=0
)
position_embedding = nn.Embedding(16, D)
position_ids = torch.arange(T)

h = (
    token_embedding(input_ids)
    + position_embedding(position_ids).unsqueeze(0)
)

# 3. 一次生成所有头需要的 Q、K、V。
q_proj = nn.Linear(D, D, bias=False)
k_proj = nn.Linear(D, D, bias=False)
v_proj = nn.Linear(D, D, bias=False)
out_proj = nn.Linear(D, D, bias=False)

q_all = q_proj(h)  # [B,T,D]
k_all = k_proj(h)
v_all = v_proj(h)


def split_heads(x, num_heads):
    batch, length, width = x.shape

    # [B,T,D]
    # -> [B,T,H,head_dim]
    # -> [B,H,T,head_dim]
    return x.reshape(
        batch, length, num_heads, width // num_heads
    ).transpose(1, 2)


# 4. 显式增加 head 维度。
query = split_heads(q_all, H)
key = split_heads(k_all, H)
value = split_heads(v_all, H)

# 5. 每个头独立计算 T×T 的分数矩阵。
scores = (
    query @ key.transpose(-2, -1)
) / math.sqrt(head_dim)

# 6. True 表示允许，为 head 维留出广播位置。
key_allowed = valid_mask[:, None, None, :]  # [B,1,1,T]

causal_allowed = torch.ones(
    T, T, dtype=torch.bool
).tril()

allowed = (
    key_allowed
    & causal_allowed[None, None, :, :]
)  # [B,1,T,T]

if not bool(allowed.any(dim=-1).all()):
    raise ValueError(
        "存在没有可读 Key 的 Query；请检查输入与 Mask"
    )

masked_scores = scores.masked_fill(
    ~allowed, float("-inf")
)
weights = torch.softmax(masked_scores, dim=-1)

head_output = weights @ value  # [B,H,T,head_dim]

# 7. 先恢复 Token 位置在前的顺序，再拼接各头。
merged = (
    head_output
    .transpose(1, 2)
    .contiguous()
    .view(B, T, D)
)

projected = out_proj(merged)

output = projected.masked_fill(
    ~valid_mask.unsqueeze(-1), 0.0
)

# 8. 观察形状与各头权重。
for name, tensor in [
    ("h", h),
    ("q_all", q_all),
    ("query", query),
    ("key", key),
    ("value", value),
    ("scores", scores),
    ("allowed", allowed),
    ("weights", weights),
    ("head_output", head_output),
    ("merged", merged),
    ("output", output),
]:
    print(name, tuple(tensor.shape))

for head in range(H):
    print(
        f"第一条输入，head {head} 权重:\n",
        weights[0, head].detach(),
    )

print(
    "每个头的权重逐行和:\n",
    weights.sum(dim=-1).detach(),
)
print(
    "PAD Query 最终输出:\n",
    output[~valid_mask].detach(),
)

# 9. 基本检查。
allowed_per_head = allowed.expand_as(weights)

assert output.shape == (B, T, D)

assert torch.allclose(
    weights.sum(-1),
    torch.ones(B, H, T),
)

assert bool(
    (
        weights.masked_select(~allowed_per_head) == 0
    ).all()
)

assert bool((output[~valid_mask] == 0).all())

assert bool(torch.isfinite(output).all())

print("基本检查通过")
```

## 8. 运行与结果

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/lesson03.py
python C02/exercise/lesson03exercise.py
```

| 对象 | 教师例子 shape |
|---|---|
| h、q_all | [2,4,8] |
| query、key、value | [2,2,4,4] |
| scores、weights | [2,2,4,4] |
| allowed | [2,1,4,4] |
| head_output | [2,2,4,4] |
| merged、output | [2,4,8] |

这些张量的尺寸碰巧相同，不代表各维含义相同。比如 Query 最后两维为“位置、特征”，scores 最后两维为“Query 位置、Key 位置”。

第一条输入的第二个 Query，在两个头的权重约为 `[0.535,0.465,0,0]` 和 `[0.452,0.548,0,0]`。第一个 Query 则都为 `[1,0,0,0]`，因为因果限制下只有一个允许的 Key。

## 9. 易错点速查

| 易错点 | 应掌握的修正 |
|---|---|
| 将原输入直接切开，认为每头只能使用一段输入 | 先投影，每个头的投影可使用完整 D 维输入 |
| 用 sqrt(D) 或 sqrt(H) 缩放 | 使用实际 Q/K 点积维度的平方根，本课为 sqrt(head_dim) |
| 沿头维做 Softmax | 沿最后的 Key 位置维做 Softmax |
| 合并时省略 transpose | 先恢复 [B,T,H,head_dim]，再拼接 |
| 给四维分数直接套三维 [B,T,T] Mask | 明确 head 广播位置，使用 [B,1,T,T]；尺寸巧合可能掩盖错误 |
| 只检查权重行和 | 另查禁止位置权重为零、有限数值、PAD 最终输出 |
| 认为头越多参数量就一定越多 | 固定 D 时，本课四个无 bias 的 D→D 投影总计 4D²；权重张量的 H 维会变化 |
| 混用接口的布尔 Mask | nn.MultiheadAttention 的布尔 Mask 中 True=屏蔽；SDPA 的布尔 attn_mask 中 True=参与 |

## 10. 本课练习与复习顺序

1. 用自己的话解释各头覆盖的 Token、投影与拆头、缩放、输出投影及因果限制。
2. 用 B=3、T=5、D=12、H=3 跟踪所有关键 shape，并解释 `weights[2,1,3,0]`。
3. 使用三条指定输入，运行 D=12 的 H=3 与 H=4 两种配置，观察 shape 与 Mask。
4. 复现 D=8、H=3 的配置错误，以及错误合并“形状相同、数值不同”的情况，说明修正依据。

作业答案与实际运行结果另见 [作业记录](C02-Lesson03-homework.md)，便于先复习再核对。

## 参考

- [Attention Is All You Need，§3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2)
- [PyTorch 2.14 MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)
- [PyTorch 2.14 Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)
- [PyTorch 2.14 scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
