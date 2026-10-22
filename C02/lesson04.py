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