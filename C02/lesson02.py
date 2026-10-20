"""C02 Lesson02：位置机制与单头 Self-Attention。

保留原有课堂代码；文末补充作业 2～4 的参考实现与检查。
正常计算不包含训练，错误实验中的 NaN 是有意展示的问题。
"""

import math
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)

# 1. 复用上一课的词表；这次观察两条输入。
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
D, d_k, d_v = 4, 2, 3

# 2. Token 向量加上可学习的绝对位置向量。
token_embedding = nn.Embedding(
    len(vocab), D, padding_idx=0, device="cpu"
)
position_embedding = nn.Embedding(16, D, device="cpu")

position_ids = torch.arange(T, device="cpu")
token_x = token_embedding(input_ids)
position_x = position_embedding(position_ids)
h = token_x + position_x.unsqueeze(0)

# 3. 同一份 h 通过三个不同的线性层。
q_proj = nn.Linear(D, d_k, bias=False, device="cpu")
k_proj = nn.Linear(D, d_k, bias=False, device="cpu")
v_proj = nn.Linear(D, d_v, bias=False, device="cpu")

query = q_proj(h)
key = k_proj(h)
value = v_proj(h)

# 4. 每一行是一个 Query，每一列是一个 Key。
scores = (
    query @ key.transpose(-2, -1)
) / math.sqrt(d_k)

# 5. 先单独观察：只禁止读取 PAD，不限制未来位置。
key_allowed = valid_mask.unsqueeze(1)  # [B,1,T]

padding_only_scores = scores.masked_fill(
    ~key_allowed, float("-inf")
)
padding_only_weights = torch.softmax(
    padding_only_scores, dim=-1
)

# 6. 再加入因果限制。
causal_allowed = torch.ones(
    T, T, dtype=torch.bool, device="cpu"
).tril()

allowed = (
    key_allowed & causal_allowed.unsqueeze(0)
)  # [B,T,T]

# 本例采用右侧补齐，每条输入至少有一个真实 Token。
assert bool(allowed.any(dim=-1).all()), (
    "存在没有可用 Key 的行，不能直接 Softmax"
)

masked_scores = scores.masked_fill(
    ~allowed, float("-inf")
)
weights = torch.softmax(masked_scores, dim=-1)

# 7. 汇总 Value，再清理 PAD 查询位置的输出。
context = weights @ value
output = context.masked_fill(
    ~valid_mask.unsqueeze(-1), 0.0
)

# 8. 观察 shape 和矩阵。
for name, tensor in [
    ("token_x", token_x),
    ("position_x", position_x),
    ("h", h),
    ("query", query),
    ("key", key),
    ("value", value),
    ("scores", scores),
    ("weights", weights),
    ("output", output),
]:
    print(name, tuple(tensor.shape))

print(
    "同一个订单，加位置前相同:",
    torch.allclose(token_x[0, 2], token_x[1, 1]),
)
print(
    "同一个订单，加位置后相同:",
    torch.allclose(h[0, 2], h[1, 1]),
)

print(
    "第一条输入只屏蔽 PAD 的权重:\n",
    padding_only_weights[0].detach(),
)
print(
    "第一条输入的允许矩阵:\n",
    allowed[0].long(),
)
print(
    "第一条输入加入因果限制后的权重:\n",
    weights[0].detach(),
)
print(
    "权重逐行求和:\n",
    weights.sum(dim=-1).detach(),
)
print(
    "被禁止位置的最大权重:",
    weights.masked_select(~allowed).max().item(),
)
print(
    "PAD 查询位置的输出:\n",
    output[~valid_mask].detach(),
)
print(
    "输出全部有限:",
    bool(torch.isfinite(output).all()),
)


# ============================================================
# 以下为作业 2～4 的参考补充（助手整理，2026-10-20）。
# 上方原有教师示例与注释保留；不把这些补充当作用户独立实现的证据。
# ============================================================


def masked_attention(attention_scores, attention_values, allowed_mask):
    """本课约定 True 表示允许；每个 Query 至少有一个可读 Key。"""
    # 在 Softmax 之前拒绝全屏蔽行，避免产生 NaN。
    if not bool(allowed_mask.any(dim=-1).all()):
        raise ValueError("存在没有可用 Key 的 Query 行；请检查全 PAD 输入或 Mask")
    masked = attention_scores.masked_fill(~allowed_mask, float("-inf"))
    attention_weights = torch.softmax(masked, dim=-1)
    return attention_weights, attention_weights @ attention_values


def run_manual_exercise():
    """第二题：计算完整矩阵，并用独立手算公式核对第二行。"""
    print("\n=== 作业第二题：手算核对 ===")
    manual_q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    manual_k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    manual_v = torch.tensor([[2., 0.], [0., 4.], [6., 6.]])
    raw = manual_q @ manual_k.transpose(-2, -1)
    scaled = raw / math.sqrt(manual_q.shape[-1])
    manual_allowed = torch.ones(3, 3, dtype=torch.bool).tril()
    manual_weights, manual_output = masked_attention(
        scaled, manual_v, manual_allowed
    )

    # 第二行保留分数 [0, 1/sqrt(2)]，第三列为未来位置。
    p0 = 1 / (1 + math.exp(1 / math.sqrt(2)))
    expected_weights = torch.tensor([p0, 1 - p0, 0.])
    expected_output = torch.tensor([2 * p0, 4 * (1 - p0)])
    matched = torch.allclose(manual_weights[1], expected_weights) and torch.allclose(
        manual_output[1], expected_output
    )

    print("原始点积分数:\n", raw)
    print("缩放后的分数:\n", scaled)
    print("屏蔽后的分数:\n", scaled.masked_fill(~manual_allowed, -torch.inf))
    print("权重:\n", manual_weights)
    print("输出:\n", manual_output)
    print("第二行与独立手算公式一致:", matched)
    return {"weights": manual_weights, "output": manual_output, "matched": matched}


def run_homework_attention():
    """第三题：从指定文本出发，独立构造 D=6、d_k=3、d_v=4 的流程。"""
    print("\n=== 作业第三题：指定输入与维度 ===")
    torch.manual_seed(42)
    mapping = {token: idx for idx, token in enumerate(vocab)}
    pad_id, unk_id = mapping["<PAD>"], mapping["<UNK>"]
    texts = ["我 查询 订单", "生成 报告", "我 查询 退款"]
    token_lists = [text.split() for text in texts]
    encoded = [[mapping.get(token, unk_id) for token in tokens] for tokens in token_lists]
    length = max(len(ids) for ids in encoded)
    padded = [ids + [pad_id] * (length - len(ids)) for ids in encoded]
    ids = torch.tensor(padded, dtype=torch.long, device="cpu")
    valid = ids.ne(pad_id)

    model_dim, key_dim, value_dim = 6, 3, 4
    token_layer = nn.Embedding(len(vocab), model_dim, padding_idx=pad_id, device="cpu")
    position_layer = nn.Embedding(16, model_dim, device="cpu")
    positions = torch.arange(length, device="cpu")
    features = token_layer(ids) + position_layer(positions).unsqueeze(0)
    queries = nn.Linear(model_dim, key_dim, bias=False, device="cpu")(features)
    keys = nn.Linear(model_dim, key_dim, bias=False, device="cpu")(features)
    values = nn.Linear(model_dim, value_dim, bias=False, device="cpu")(features)
    attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(key_dim)

    readable_keys = valid.unsqueeze(1)
    padding_weights, _ = masked_attention(attention_scores, values, readable_keys)
    causal = torch.ones(length, length, dtype=torch.bool, device="cpu").tril()
    readable = readable_keys & causal.unsqueeze(0)
    attention_weights, mixed = masked_attention(attention_scores, values, readable)
    result = mixed.masked_fill(~valid.unsqueeze(-1), 0.0)

    print("input_ids:\n", ids)
    print("valid_mask:\n", valid.long())
    for name, tensor in [
        ("h", features), ("Q", queries), ("K", keys), ("Value", values),
        ("scores", attention_scores), ("weights", attention_weights), ("output", result),
    ]:
        print(name, tuple(tensor.shape))
    print("第二条输入只屏蔽 PAD 的权重:\n", padding_weights[1].detach())
    print("第二条输入同时限制未来的权重:\n", attention_weights[1].detach())
    print("PAD 查询位置输出:\n", result[~valid].detach())
    return {
        "input_ids": ids, "valid": valid, "query": queries, "key": keys,
        "value": values, "scores": attention_scores, "padding_weights": padding_weights,
        "allowed": readable, "weights": attention_weights, "context": mixed, "output": result,
    }


def run_error_experiments(teacher_scores, teacher_values, teacher_allowed):
    """第四题：每次只改变一个因素，故意展示错误，再由正确流程对照。"""
    print("\n=== 作业第四题：三个错误实验 ===")

    # 错误一：将禁止位置填成 0。行和仍是 1，但禁止位置会分到权重。
    zero_filled = teacher_scores.masked_fill(~teacher_allowed, 0.0)
    zero_weights = torch.softmax(zero_filled, dim=-1)
    forbidden_max = zero_weights.masked_select(~teacher_allowed).max().detach().item()
    print("错误一，禁止位置最大权重:", forbidden_max)
    print("错误一，权重逐行和:\n", zero_weights.sum(-1).detach())

    # 错误二：沿 Query 维度归一化。本例 PAD 列全是 -inf，产生 NaN。
    correctly_masked = teacher_scores.masked_fill(~teacher_allowed, -torch.inf)
    wrong_axis_weights = torch.softmax(correctly_masked, dim=-2)
    wrong_axis_finite = bool(torch.isfinite(wrong_axis_weights).all())
    print("错误二，权重全部有限:", wrong_axis_finite)
    print("错误二，权重逐行和:\n", wrong_axis_weights.sum(-1).detach())

    # 错误三：一条输入全部是 PAD。故意先演示直接 Softmax 的结果。
    length = teacher_scores.shape[-1]
    all_pad_ids = torch.zeros((1, length), dtype=torch.long, device="cpu")
    all_pad_valid = all_pad_ids.ne(0)
    causal = torch.ones(length, length, dtype=torch.bool, device="cpu").tril()
    all_pad_allowed = all_pad_valid.unsqueeze(1) & causal.unsqueeze(0)
    empty_scores = torch.zeros_like(teacher_scores[:1])
    unsafe_weights = torch.softmax(empty_scores.masked_fill(~all_pad_allowed, -torch.inf), -1)
    all_pad_is_nan = bool(torch.isnan(unsafe_weights).all())
    print("错误三，直接 Softmax 得到全 NaN:", all_pad_is_nan)

    guard_triggered = False
    try:
        masked_attention(empty_scores, teacher_values[:1], all_pad_allowed)
    except ValueError as exc:
        guard_triggered = True
        print("正确保护已触发（预期 ValueError）:", str(exc))
    else:
        raise AssertionError("全 PAD 样本应在 Softmax 前被拒绝")

    return {
        "zero_fill_forbidden_max": forbidden_max,
        "wrong_axis_all_finite": wrong_axis_finite,
        "all_pad_direct_softmax_all_nan": all_pad_is_nan,
        "all_pad_guard_triggered": guard_triggered,
    }


if __name__ == "__main__":
    manual_result = run_manual_exercise()
    homework = run_homework_attention()
    error_results = run_error_experiments(scores, value, allowed)

    # 检查正常计算结果与有意触发的错误现象；NaN 仅出现在错误演示中。
    homework_weights = homework["weights"]
    homework_output = homework["output"]
    checks = {
        "teacher_output_shape": tuple(output.shape) == (2, 4, 3),
        "teacher_weights_normalized": torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1))),
        "teacher_forbidden_zero": bool((weights.masked_select(~allowed) == 0).all()),
        "teacher_output_finite": bool(torch.isfinite(output).all()),
        "manual_second_row": manual_result["matched"],
        "homework_input_ids": homework["input_ids"].tolist() == [[2, 3, 4], [5, 6, 0], [2, 3, 7]],
        "homework_feature_shapes": tuple(homework["query"].shape) == (3, 3, 3) and tuple(homework["key"].shape) == (3, 3, 3) and tuple(homework["value"].shape) == (3, 3, 4),
        "homework_output_shape": tuple(homework_output.shape) == (3, 3, 4),
        "homework_weights_normalized": torch.allclose(homework_weights.sum(-1), torch.ones_like(homework_weights.sum(-1))),
        "homework_forbidden_zero": bool((homework_weights.masked_select(~homework["allowed"]) == 0).all()),
        "homework_output_finite": bool(torch.isfinite(homework_output).all()),
        "homework_pad_output_zero": bool((homework_output[~homework["valid"]] == 0).all()),
        "zero_fill_error_reproduced": error_results["zero_fill_forbidden_max"] > 0,
        "wrong_axis_error_reproduced": not error_results["wrong_axis_all_finite"],
        "all_pad_error_and_guard": error_results["all_pad_direct_softmax_all_nan"] and error_results["all_pad_guard_triggered"],
    }
    print("\n=== 本课检查 ===")
    for name, passed in checks.items():
        print(name, passed)
    assert all(checks.values()), "存在未通过的检查"
    print(f"检查通过：{sum(checks.values())}/{len(checks)}")
