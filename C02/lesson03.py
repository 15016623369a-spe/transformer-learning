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