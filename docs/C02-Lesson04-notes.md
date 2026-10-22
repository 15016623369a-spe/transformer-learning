# C02 Lesson04：FFN、残差、LayerNorm 与 Transformer Block

整理日期：2026-10-22（Asia/Shanghai）。[课堂脚本](../C02/lesson04.py) 保留本次检查时的原有内容；[作业参考](../C02/exercise/lesson04exercise.py) 为本次明确标识的助手补充。

## 优先掌握的五个重点

1. **Attention 混合允许访问的位置；FFN 分别加工每个位置，参数跨位置共享。** FFN 输入可能已包含上下文，但 FFN 本身不跨 Token 混合。
2. **FFN 的形状为 D → D_ff → D。** 改变 D_ff 只改变中间特征宽度，B、T 和最终 D 不变；激活函数提供非线性。
3. **两次残差相加都保留原旁路，且发生在 [B,T,D] 上。** 不要把归一化后的向量替换到旁路，也不要直接加 FFN 中间层。
4. **nn.LayerNorm(D) 对每个 Token 的 D 个特征独立计算均值和方差。** 方差分母为 D，手算对应 correction=0；LayerNorm 不产生概率。
5. **本课实现 Pre-Norm：先归一化，再进 Attention/FFN，最后加残差。** 继续保留因果限制、PAD Key 屏蔽、全屏蔽行检查，以及残差之后的 PAD 清零。

## 1. 从多头注意力到完整 Block

Lesson03 得到的是 Attention 分支结果。本课把它放进残差结构，并添加 LayerNorm 和 FFN：

```text
x ───────────────────────┐
└→ LN1 → Attention ──────+ → u ─────────────────────┐
                            └→ LN2 → FFN ──────────+ → y
```

```python
u = x + attention(norm1(x))
y = u + ffn(norm2(u))
```

norm1 与 norm2 是两组独立的参数。这里省略 Dropout 和 PAD 清零以展示计算顺序；完整代码会在两次残差之后清零 PAD。此处实现一个使用因果自注意力的 Pre-Norm Block，不含 Cross-Attention、词表输出层或训练循环。

## 2. FFN 与激活函数

```python
ff_hidden = ff_up(normed2)      # [B,T,D_ff]
ff_active = activation(ff_hidden)
ff_out = ff_down(ff_active)    # [B,T,D]
```

Linear 只作用于最后一维，相同的两层参数用于所有位置。课堂配置 D=8、D_ff=16，形状依次为 `[2,4,8] → [2,4,16] → [2,4,8]`。

没有激活函数时，连续两个仿射变换可合并为一个：`(xW1+b1)W2+b2 = x(W1W2)+(b1W2+b2)`。本课在两层之间使用 GELU，即 `x * Φ(x)`，输出形状不变。[PyTorch GELU 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html)

## 3. 残差连接

```python
after_attn = clear_pad(x + attn_out, valid_mask)
output = clear_pad(after_attn + ff_out, valid_mask)
```

旁路保留输入，分支学习对表示的调整。它也为梯度传播提供直接路径，但不表示加入残差后任意模型都能自动稳定训练。

**必须比较实际参与相加的张量。** ff_hidden 仍为 D_ff 维，完整 FFN 输出才恢复到 D 维。本课不要依赖意外广播、裁剪或补零来掩盖维度错误。

## 4. LayerNorm 的统计范围与公式

对固定 b、t，令 x 表示该 Token 的 D 个特征：

\[
\mu=\frac{1}{D}\sum_{d=1}^{D}x_d,\qquad
\sigma^2=\frac{1}{D}\sum_{d=1}^{D}(x_d-\mu)^2
\]

\[
y_d=\gamma_d\frac{x_d-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_d
\]

```python
mean = x.mean(dim=-1, keepdim=True)
variance = x.var(dim=-1, keepdim=True, correction=0)
normalized = (x - mean) / torch.sqrt(variance + eps)
```

gamma、beta 按特征学习，分别初始化为 1、0；手算完整输出还需乘 gamma、加 beta。默认 eps=1e-5。eps 会使归一化方差通常略小于 1；训练后的仿射参数也会改变均值和方差，所以不能始终要求最终输出严格均值 0、方差 1。

`nn.LayerNorm(D)` 的 D 指输入末尾的特征维，**不是序列长度 T**；统计不会跨 B 或 T 混合。[PyTorch LayerNorm 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)

## 5. Pre-Norm 与 Post-Norm

| 顺序 | Attention 子层 | FFN 子层 |
|---|---|---|
| Pre-Norm，本课采用 | u = x + Attention(LN1(x)) | y = u + FFN(LN2(u)) |
| Post-Norm | u = LN1(x + Attention(x)) | y = LN2(u + FFN(u)) |

PyTorch TransformerEncoderLayer 的 norm_first=True 对应先归一化的顺序。本次对照同时设置 batch_first=True、dropout=0、activation="gelu" 并复制参数；对照输出也按本课约定将 PAD 清零。[PyTorch TransformerEncoderLayer 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)

## 6. Mask 与 PAD 沿用规则

```python
allowed = valid_mask[:, None, None, :] & causal[None, None, :, :]
weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
```

本课 allowed 的 True 表示允许；调用原生 TransformerEncoderLayer 的布尔 Mask 时需要转换为 True=禁止的语义，不能直接照搬。

本例输入非空、采用右侧 PAD。PAD Query 内部可读取前面的真实 Key，其输出随后清零；PAD Key 始终禁止被读取。若某行没有可读 Key，代码先抛出 ValueError，避免全为负无穷的 Softmax 行产生 NaN。本写法不声称支持左侧补齐或全 PAD 句子。

即便 PAD 的输入为零，带 bias 的 FFN 仍可能输出非零；LayerNorm 的 beta 训练后也可能非零。因此 embedding 的 padding_idx=0 不代替残差后的清零。masked_scores 中被屏蔽位置的 -inf 是有意设置，有限值检查不把它误判为正常输出错误。

## 7. 完整教师代码

下列内容与本次检查的课堂文件一致，原有计算未改动。

```python
import math
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)

# 1. 输入与维度。
vocab = [
    "<PAD>", "<UNK>", "我", "查询", "订单",
    "生成", "报告", "退款", "分析",
]

input_ids = torch.tensor(
    [[2, 3, 4, 0], [5, 4, 8, 6]],
    dtype=torch.long,
)

valid = input_ids.ne(0)
B, T = input_ids.shape

D, H, D_ff = 8, 2, 16

if H <= 0 or D % H != 0:
    raise ValueError(
        "H 必须为正整数，且 D 必须能被 H 整除"
    )

head_dim = D // H

token_embedding = nn.Embedding(
    len(vocab), D, padding_idx=0
)
position_embedding = nn.Embedding(16, D)

# 2. 在函数外建立层。
# 多次调用 run_block 时，继续使用同一套参数。
norm1 = nn.LayerNorm(D)
norm2 = nn.LayerNorm(D)

q_proj = nn.Linear(D, D, bias=False)
k_proj = nn.Linear(D, D, bias=False)
v_proj = nn.Linear(D, D, bias=False)
out_proj = nn.Linear(D, D, bias=False)

ff_up = nn.Linear(D, D_ff)
activation = nn.GELU()
ff_down = nn.Linear(D_ff, D)


def clear_pad(x, valid_mask):
    return x.masked_fill(
        ~valid_mask.unsqueeze(-1), 0.0
    )


def run_block(x, valid_mask):
    batch, length, width = x.shape

    # 3. 第一条分支：归一化后计算 Attention。
    normed1 = norm1(x)

    def split_heads(tensor):
        return tensor.reshape(
            batch, length, H, head_dim
        ).transpose(1, 2)

    query = split_heads(q_proj(normed1))
    key = split_heads(k_proj(normed1))
    value = split_heads(v_proj(normed1))

    scores = (
        query @ key.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    causal = torch.ones(
        length, length, dtype=torch.bool
    ).tril()

    allowed = (
        valid_mask[:, None, None, :]
        & causal[None, None, :, :]
    )

    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("存在没有可读 Key 的 Query")

    masked_scores = scores.masked_fill(
        ~allowed, -torch.inf
    )
    weights = torch.softmax(masked_scores, dim=-1)

    head_output = weights @ value

    merged = (
        head_output
        .transpose(1, 2)
        .contiguous()
        .view(batch, length, width)
    )

    attn_out = out_proj(merged)

    # 4. 第一次残差相加：旁路保留原 x。
    after_attn = clear_pad(
        x + attn_out, valid_mask
    )

    # 5. 第二条分支：归一化后计算 FFN。
    normed2 = norm2(after_attn)

    ff_hidden = ff_up(normed2)
    ff_active = activation(ff_hidden)
    ff_out = ff_down(ff_active)

    # 6. 第二次残差相加。
    output = clear_pad(
        after_attn + ff_out, valid_mask
    )

    trace = {
        "normed1": normed1,
        "query": query,
        "scores": scores,
        "allowed": allowed,
        "weights": weights,
        "attn_out": attn_out,
        "after_attn": after_attn,
        "normed2": normed2,
        "ff_hidden": ff_hidden,
        "ff_active": ff_active,
        "ff_out": ff_out,
        "output": output,
    }

    return output, trace


# 7. Token 向量加位置向量。
position_ids = torch.arange(T)

h = (
    token_embedding(input_ids)
    + position_embedding(position_ids).unsqueeze(0)
)

# 本课约定：PAD 位置在 Block 输入及各次残差后清零。
h = clear_pad(h, valid)

output, trace = run_block(h, valid)

# 8. 观察中间结果。
print("input_ids:\n", input_ids)
print("h", tuple(h.shape))

for name, tensor in trace.items():
    print(name, tuple(tensor.shape))

print(
    "Norm1 后真实 Token 的均值:\n",
    trace["normed1"].mean(-1)[valid].detach(),
)
print(
    "Norm1 后真实 Token 的方差:\n",
    trace["normed1"].var(
        -1, correction=0
    )[valid].detach(),
)
print(
    "PAD 位置的 FFN 分支输出:\n",
    trace["ff_out"][~valid].detach(),
)
print(
    "PAD 位置的最终输出:\n",
    output[~valid].detach(),
)

# 9. 用手算公式核对初始 LayerNorm。
# 此时 gamma=1、beta=0，尚未训练。
mean = h.mean(dim=-1, keepdim=True)
variance = h.var(
    dim=-1, keepdim=True, correction=0
)
manual_norm = (
    (h - mean) / torch.sqrt(variance + norm1.eps)
)

allowed_per_head = trace["allowed"].expand_as(
    trace["weights"]
)

checks = {
    "shape_preserved": output.shape == h.shape,

    "ffn_expanded": (
        trace["ff_hidden"].shape == (B, T, D_ff)
    ),

    "manual_layernorm_matches": torch.allclose(
        trace["normed1"], manual_norm, atol=1e-6
    ),

    "attention_rows_sum_one": torch.allclose(
        trace["weights"].sum(-1),
        torch.ones(B, H, T),
    ),

    "forbidden_weights_zero": bool(
        (
            trace["weights"].masked_select(
                ~allowed_per_head
            ) == 0
        ).all()
    ),

    "pad_output_zero": bool(
        (output[~valid] == 0).all()
    ),

    "normal_values_finite": all(
        bool(torch.isfinite(t).all())
        for t in trace.values()
    ),
}

print("检查:", checks)
assert all(checks.values())

# 10. 单独观察四维向量的 LayerNorm。
demo = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(
    "LayerNorm 小例子:",
    nn.LayerNorm(
        4, elementwise_affine=False
    )(demo),
)
```

## 8. 易错点与检查方式

| 易错点 | 应如何检查或修正 |
|---|---|
| 将 FFN 当成跨 Token 的模块 | 它只处理末尾特征维，参数在各位置共享 |
| 两层 Linear 中间没有激活 | 加入 GELU，避免退化为单次仿射变换 |
| 使用 x 的归一化结果作为 Pre-Norm 旁路 | 旁路仍保留 x；归一化结果仅进入分支 |
| after_attn + ff_hidden | 先完成激活与 D_ff→D 的投影 |
| LayerNorm(T) | 本课按每个 Token 的特征做归一化，使用 LayerNorm(D) |
| 手算直接用默认 x.var() | 明确 correction=0，并加 eps |
| 把 PAD Query 整行屏蔽后直接 Softmax | 先保证每行有可读 Key，PAD 输出再清零 |
| 忘记 FFN bias 可能让 PAD 再次非零 | 两次残差后均清零 PAD |
| 每次调用 run_block 都重新创建层 | 层在函数外创建，多次前向使用同一套参数 |

## 9. 作业与复核入口

四题的题意、六个概念答案、LayerNorm 手算、两种 FFN 宽度、两个错误案例见 [作业答案与检查](C02-Lesson04-homework.md)。运行证据见 [验证记录](C02-Lesson04-verification.json)。

```bash
conda activate fde
cd /Users/jace/transformer-learning
python C02/lesson04.py
python C02/exercise/lesson04exercise.py
```
