"""C02 Lesson04 助手参考补充：LayerNorm 手算、FFN 宽度比较、错误修正。

本文件整理自聊天中的参考答案；不是用户原有练习或独立完成的证据。
采用非空输入、右侧 PAD、CPU 前向计算，不训练。
"""

import math
import torch
from torch import nn

torch.set_printoptions(precision=6, sci_mode=False)


def layernorm_exercise():
    x = torch.tensor([0.0, 2.0, 2.0, 4.0])
    eps = 1e-5
    mean = x.mean()
    variance = x.var(correction=0)
    manual = (x - mean) / torch.sqrt(variance + eps)
    actual = nn.LayerNorm(4, eps=eps, elementwise_affine=False)(x)
    print("第二题均值:", mean.item())
    print("第二题方差:", variance.item())
    print("手算公式:", manual)
    print("LayerNorm:", actual)
    assert torch.allclose(manual, actual, atol=1e-6)
    return {'mean': mean.item(), 'variance': variance.item(), 'output': actual.tolist(), 'matches': True}


def run_experiment(D_ff):
    torch.manual_seed(42)
    D, H = 12, 3
    head_dim = D // H
    vocab = [
        "<PAD>", "<UNK>", "我", "查询", "订单",
        "生成", "报告", "退款", "分析",
    ]
    mapping = {token: i for i, token in enumerate(vocab)}
    texts = ["我 查询 订单", "生成 报告", "我 查询 退款"]
    encoded = [[mapping.get(token, mapping['<UNK>']) for token in text.split()] for text in texts]
    T = max(len(ids) for ids in encoded)
    ids = torch.tensor([row + [0] * (T - len(row)) for row in encoded], dtype=torch.long)
    valid = ids.ne(0)
    B = ids.shape[0]

    token_embedding = nn.Embedding(len(vocab), D, padding_idx=0)
    position_embedding = nn.Embedding(16, D)
    norm1, norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
    q_proj = nn.Linear(D, D, bias=False)
    k_proj = nn.Linear(D, D, bias=False)
    v_proj = nn.Linear(D, D, bias=False)
    out_proj = nn.Linear(D, D, bias=False)
    ff_up = nn.Linear(D, D_ff)
    activation = nn.GELU()
    ff_down = nn.Linear(D_ff, D)

    def clear_pad(x):
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)

    h = clear_pad(token_embedding(ids) + position_embedding(torch.arange(T)).unsqueeze(0))

    # 第一部分：x + Attention(norm1(x))。
    normed1 = norm1(h)

    def split_heads(x):
        return x.reshape(B, T, H, head_dim).transpose(1, 2)

    query = split_heads(q_proj(normed1))
    key = split_heads(k_proj(normed1))
    value = split_heads(v_proj(normed1))
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    allowed = valid[:, None, None, :] & causal[None, None, :, :]
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("存在没有可读 Key 的 Query")
    weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    merged = (weights @ value).transpose(1, 2).contiguous().view(B, T, D)
    attn_out = out_proj(merged)
    after_attn = clear_pad(h + attn_out)

    # 第二部分：u + FFN(norm2(u))。
    normed2 = norm2(after_attn)
    ff_hidden = ff_up(normed2)
    ff_active = activation(ff_hidden)
    ff_out = ff_down(ff_active)
    output = clear_pad(after_attn + ff_out)

    print(f"\n=== D_ff={D_ff} ===")
    print("input_ids:\n", ids)
    for name, tensor in [
        ('h', h), ('normed1', normed1), ('query', query), ('scores', scores),
        ('allowed', allowed), ('weights', weights), ('attn_out', attn_out),
        ('after_attn', after_attn), ('normed2', normed2),
        ('ff_hidden', ff_hidden), ('ff_active', ff_active), ('ff_out', ff_out), ('output', output),
    ]:
        print(name, tuple(tensor.shape))
    print("PAD 最终输出:\n", output[~valid].detach())

    checks = {
        'residual_shapes': h.shape == attn_out.shape == after_attn.shape == ff_out.shape == output.shape,
        'ffn_hidden_shape': ff_hidden.shape == (B, T, D_ff),
        'output_shape': output.shape == (B, T, D),
        'attention_rows_sum_one': torch.allclose(weights.sum(-1), torch.ones(B, H, T)),
        'forbidden_weights_zero': bool((weights.masked_select(~allowed.expand_as(weights)) == 0).all()),
        'pad_output_zero': bool((output[~valid] == 0).all()),
        'normal_values_finite': all(bool(torch.isfinite(x).all()) for x in [h, normed1, query, key, value, weights, attn_out, after_attn, normed2, ff_hidden, ff_active, ff_out, output]),
    }
    print('检查:', checks)
    assert all(checks.values())
    return {
        'D': D, 'H': H, 'D_ff': D_ff, 'h': h, 'valid': valid, 'causal': causal,
        'norm1': norm1, 'norm2': norm2, 'q_proj': q_proj, 'k_proj': k_proj,
        'v_proj': v_proj, 'out_proj': out_proj, 'ff_up': ff_up, 'ff_down': ff_down,
        'output': output, 'after_attn': after_attn, 'checks': checks,
    }


def debug_exercises():
    torch.manual_seed(42)
    B, T, D, D_ff = 2, 4, 8, 16
    after_attn = torch.randn(B, T, D)
    norm2 = nn.LayerNorm(D)
    ff_up = nn.Linear(D, D_ff)
    activation = nn.GELU()
    ff_down = nn.Linear(D_ff, D)
    ff_hidden = ff_up(norm2(after_attn))

    errors = {}
    try:
        bad_output = after_attn + ff_hidden
    except RuntimeError as error:
        errors['A'] = str(error)
        print('\n案例 A 预期报错:', error)
    ff_out = ff_down(activation(ff_hidden))
    corrected_output = after_attn + ff_out
    assert corrected_output.shape == after_attn.shape

    h = torch.randn(B, T, D)
    try:
        bad_result = nn.LayerNorm(T)(h)
    except RuntimeError as error:
        errors['B'] = str(error)
        print('案例 B 预期报错:', error)
    corrected_norm = nn.LayerNorm(D)(h)
    assert corrected_norm.shape == h.shape
    assert set(errors) == {'A', 'B'}
    print('两个案例修正后的 shape:', corrected_output.shape, corrected_norm.shape)
    return errors


if __name__ == '__main__':
    layernorm_result = layernorm_exercise()
    result_24 = run_experiment(24)
    result_48 = run_experiment(48)
    expected_errors = debug_exercises()
