import math
import torch
from torch import nn

torch.set_printoptions(precision=3, sci_mode=False)


def run_attention(H):
    # 两次运行使用相同随机种子。
    torch.manual_seed(42)

    D = 12
    if H <= 0 or D % H != 0:
        raise ValueError(
            "H 必须为正整数，且 D 必须能被 H 整除"
        )

    head_dim = D // H

    # 1. 编码并右侧补齐。
    vocab = [
        "<PAD>", "<UNK>", "我", "查询", "订单",
        "生成", "报告", "退款", "分析",
    ]
    token_to_id = {
        token: i for i, token in enumerate(vocab)
    }

    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]

    texts = [
        "我 查询 订单",
        "生成 报告",
        "我 查询 退款",
    ]

    encoded = [
        [
            token_to_id.get(token, unk_id)
            for token in text.split()
        ]
        for text in texts
    ]

    T = max(len(ids) for ids in encoded)

    input_ids = torch.tensor(
        [
            ids + [pad_id] * (T - len(ids))
            for ids in encoded
        ],
        dtype=torch.long,
    )

    valid = input_ids.ne(pad_id)
    B = input_ids.shape[0]

    # 2. Token 向量与位置向量。
    token_embedding = nn.Embedding(
        len(vocab), D, padding_idx=pad_id
    )
    position_embedding = nn.Embedding(16, D)

    h = (
        token_embedding(input_ids)
        + position_embedding(torch.arange(T)).unsqueeze(0)
    )

    # 3. Q、K、V 与输出投影。
    q_proj = nn.Linear(D, D, bias=False)
    k_proj = nn.Linear(D, D, bias=False)
    v_proj = nn.Linear(D, D, bias=False)
    out_proj = nn.Linear(D, D, bias=False)

    q_all = q_proj(h)
    k_all = k_proj(h)
    v_all = v_proj(h)

    # 4. 拆头。
    def split_heads(x):
        return x.reshape(
            B, T, H, head_dim
        ).transpose(1, 2)

    query = split_heads(q_all)
    key = split_heads(k_all)
    value = split_heads(v_all)

    # 5. 各头分别计算分数。
    scores = (
        query @ key.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    # 6. Padding Mask 与 Causal Mask。
    key_allowed = valid[:, None, None, :]

    causal = torch.ones(
        T, T, dtype=torch.bool
    ).tril()

    allowed = (
        key_allowed & causal[None, None, :, :]
    )

    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("存在没有可读 Key 的 Query")

    masked_scores = scores.masked_fill(
        ~allowed, -torch.inf
    )
    weights = torch.softmax(masked_scores, dim=-1)

    # 7. 汇总 Value，合并各头，进行输出投影。
    head_output = weights @ value

    merged = (
        head_output
        .transpose(1, 2)
        .contiguous()
        .view(B, T, D)
    )

    output = out_proj(merged).masked_fill(
        ~valid.unsqueeze(-1), 0.0
    )

    # 8. 打印关键结果。
    print(f"\n=== H={H}, head_dim={head_dim} ===")
    print("input_ids:\n", input_ids)

    for name, tensor in [
        ("h", h),
        ("q_all", q_all),
        ("k_all", k_all),
        ("v_all", v_all),
        ("query", query),
        ("key", key),
        ("value", value),
        ("scores", scores),
        ("key_allowed", key_allowed),
        ("allowed", allowed),
        ("weights", weights),
        ("head_output", head_output),
        ("merged", merged),
        ("output", output),
    ]:
        print(name, tuple(tensor.shape))

    for head in range(H):
        print(
            f"第二条输入 head {head}:\n",
            weights[1, head].detach(),
        )

    # 9. 检查。
    checks = {
        "output_shape": output.shape == (B, T, D),

        "row_sums_one": torch.allclose(
            weights.sum(-1),
            torch.ones(B, H, T),
        ),

        "forbidden_weights_zero": bool(
            (
                weights.masked_select(
                    ~allowed.expand_as(weights)
                ) == 0
            ).all()
        ),

        "pad_query_output_zero": bool(
            (output[~valid] == 0).all()
        ),

        "weights_finite": bool(
            torch.isfinite(weights).all()
        ),

        "output_finite": bool(
            torch.isfinite(output).all()
        ),
    }

    print("检查:", checks)
    assert all(checks.values())

    # 参考补充（助手整理，2026-10-22）：返回中间结果，供原生实现对照。
    # 上方原有计算、检查和学习注释保留。
    return {
        "D": D, "H": H, "input_ids": input_ids, "valid": valid,
        "h": h, "query": query, "key": key, "value": value,
        "allowed": allowed, "weights": weights, "output": output,
        "causal": causal, "q_proj": q_proj, "k_proj": k_proj,
        "v_proj": v_proj, "out_proj": out_proj, "checks": checks,
    }



# ============================================================
# 以下为第四题参考补充（助手整理，2026-10-22）。
# 用户原文件已包含第三题的两组配置；此处补齐错误复现与修正。
# ============================================================


def run_debug_exercises():
    print("\n=== 第四题：案例 A，等宽拆头的配置检查 ===")
    model_dim, heads = 8, 3
    rejected = False
    try:
        if heads <= 0 or model_dim % heads != 0:
            raise ValueError("D 不能被 H 整除，无法使用本课的等宽拆头")
    except ValueError as error:
        rejected = True
        print("预期报错:", error)

    corrected_heads = 2
    assert model_dim % corrected_heads == 0
    print("保留 D=8，修正为 H=2，每个头维度:", model_dim // corrected_heads)

    print("\n=== 第四题：案例 B，形状正确但数值顺序错误 ===")
    toy = torch.arange(48).reshape(2, 3, 8)
    split = toy.reshape(2, 3, 2, 4).transpose(1, 2)
    bad_restored = split.reshape(2, 3, 8)
    restored = split.transpose(1, 2).contiguous().view(2, 3, 8)

    checks = {
        "invalid_head_config_rejected": rejected,
        "wrong_merge_shape_same": bad_restored.shape == toy.shape,
        "wrong_merge_values_differ": not torch.equal(bad_restored, toy),
        "correct_merge_restores_values": torch.equal(restored, toy),
    }
    print("原始第一行:", toy[0, 0])
    print("错误合并第一行:", bad_restored[0, 0])
    print("错误与修正检查:", checks)
    assert all(checks.values())
    return checks


if __name__ == "__main__":
    case_h3 = run_attention(3)
    case_h4 = run_attention(4)
    debug_checks = run_debug_exercises()
