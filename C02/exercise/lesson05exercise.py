"""C02 Lesson05 助手参考补充（2026-10-22）。

原课堂文件 lesson05.py 保留不变；此处补齐作业配置、对照实验、别名问题和全屏蔽行修正。
本文件为聊天答案的参考实现，不是用户原有独立作业或独立掌握的证据。
"""

import math
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)


def clear_pad(x, valid):
    return x.masked_fill(~valid.unsqueeze(-1), 0.0)


class TinyPreNormBlock(nn.Module):
    def __init__(self, D, H, D_ff):
        super().__init__()
        if H <= 0 or D % H != 0:
            raise ValueError("H 必须为正数，且 D 能被 H 整除")
        self.H = H
        self.head_dim = D // H
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        self.ff_up = nn.Linear(D, D_ff)
        self.activation = nn.GELU()
        self.ff_down = nn.Linear(D_ff, D)

    def forward(self, x, valid, use_causal=True, mask_pad_keys=True):
        B, T, D = x.shape
        normed1 = self.norm1(x)

        def split_heads(tensor):
            return tensor.reshape(
                B, T, self.H, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(normed1))
        key = split_heads(self.k_proj(normed1))
        value = split_heads(self.v_proj(normed1))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 两个开关用于课堂对照，正常调用时都保持 True。
        allowed = torch.ones(B, 1, T, T, dtype=torch.bool, device=x.device)
        if mask_pad_keys:
            allowed = allowed & valid[:, None, None, :]
        if use_causal:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            allowed = allowed & causal[None, None, :, :]
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("存在没有可读 Key 的 Query")

        weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
        head_output = weights @ value
        merged = head_output.transpose(1, 2).contiguous().view(B, T, D)
        after_attn = clear_pad(x + self.out_proj(merged), valid)

        ff_hidden = self.ff_up(self.norm2(after_attn))
        ff_out = self.ff_down(self.activation(ff_hidden))
        output = clear_pad(after_attn + ff_out, valid)
        return output, {
            "query": query,
            "weights": weights,
            "allowed": allowed,
            "after_attn": after_attn,
            "ff_hidden": ff_hidden,
            "output": output,
        }



def run_homework():
    torch.manual_seed(42)
    input_ids = torch.tensor(
        [[2, 3, 4, 0], [5, 6, 0, 0], [2, 3, 7, 8]],
        dtype=torch.long,
    )
    valid = input_ids.ne(0)
    B, T = input_ids.shape
    D, H, D_ff = 12, 3, 24
    token_embedding = nn.Embedding(9, D, padding_idx=0)
    position_embedding = nn.Embedding(16, D)
    block = TinyPreNormBlock(D, H, D_ff)
    block.eval()

    with torch.no_grad():
        h = token_embedding(input_ids)
        h = h + position_embedding(torch.arange(T)).unsqueeze(0)
        h = clear_pad(h, valid)
        output, trace = block(h, valid)
        shapes = {
            'input_ids': list(input_ids.shape), 'h': list(h.shape),
            **{name: list(trace[name].shape) for name in (
                'query', 'weights', 'allowed', 'after_attn', 'ff_hidden', 'output'
            )},
        }
        forbidden = ~trace['allowed'].expand_as(trace['weights'])
        checks = {
            'output_shape': output.shape == (B, T, D),
            'ffn_hidden_shape': trace['ff_hidden'].shape == (B, T, D_ff),
            'weights_rows_sum_one': torch.allclose(trace['weights'].sum(-1), torch.ones(B, H, T)),
            'forbidden_weights_zero': bool((trace['weights'][forbidden] == 0).all()),
            'pad_output_zero': bool((output[~valid] == 0).all()),
            'normal_values_finite': all(bool(torch.isfinite(tensor).all()) for tensor in [h, *trace.values()]),
        }
        assert all(checks.values())

        # 第三题 A：第三条输入最后一个真实位置被扰动。
        changed_future = h.clone()
        changed_future[2, -1] += torch.linspace(-2.0, 2.0, D)
        causal_changed, _ = block(changed_future, valid)
        torch.testing.assert_close(causal_changed[2, :-1], output[2, :-1], atol=1e-6, rtol=1e-5)
        causal_delta = (causal_changed[2, :-1] - output[2, :-1]).abs().max().item()

        open_base, _ = block(h, valid, use_causal=False)
        open_changed, _ = block(changed_future, valid, use_causal=False)
        open_delta = (open_changed[2, :-1] - open_base[2, :-1]).abs().max().item()
        assert not torch.allclose(open_changed[2, :-1], open_base[2, :-1], atol=1e-6, rtol=1e-5)

        # 第三题 B：关闭因果限制，单独检验 PAD Key 屏蔽。
        changed_pad = h.clone()
        changed_pad[~valid] = torch.arange(1, D + 1, dtype=h.dtype) * 10.0
        pad_changed, _ = block(changed_pad, valid, use_causal=False)
        torch.testing.assert_close(pad_changed[valid], open_base[valid], atol=1e-6, rtol=1e-5)
        pad_delta = (pad_changed[valid] - open_base[valid]).abs().max().item()

        unmasked_base, _ = block(h, valid, use_causal=False, mask_pad_keys=False)
        unmasked_changed, _ = block(changed_pad, valid, use_causal=False, mask_pad_keys=False)
        unmasked_delta = (unmasked_changed[valid] - unmasked_base[valid]).abs().max().item()
        assert not torch.allclose(unmasked_changed[valid], unmasked_base[valid], atol=1e-6, rtol=1e-5)

    deltas = {
        'causal_prefix_max_delta': causal_delta,
        'without_causal_prefix_max_delta': open_delta,
        'pad_mask_valid_output_max_delta': pad_delta,
        'without_pad_mask_valid_output_max_delta': unmasked_delta,
    }
    print('第二题形状:', shapes)
    print('第二题检查:', checks)
    print('第三题对照实验:', deltas)
    return {
        'input_ids': input_ids, 'valid': valid, 'h': h, 'output': output,
        'D': D, 'H': H, 'D_ff': D_ff, 'block': block,
        'shapes': shapes, 'checks': checks, 'deltas': deltas,
    }


def debug_exercises(state):
    h = state['h']
    valid = state['valid']
    block = state['block']
    D = state['D']
    original_h = h.clone()

    # A：用局部副本复现别名问题，避免污染前面实验的 h。
    bad_h = h.clone()
    before = bad_h.clone()
    changed = bad_h
    changed[2, -1] += torch.linspace(-2.0, 2.0, D)
    alias_same_object = changed is bad_h
    baseline_corrupted = not torch.equal(bad_h, before)
    assert alias_same_object and baseline_corrupted

    # 修正：实验副本与基线独立。
    good_h = h.clone()
    before_good = good_h.clone()
    changed_good = good_h.clone()
    changed_good[2, -1] += torch.linspace(-2.0, 2.0, D)
    assert torch.equal(good_h, before_good)
    assert not torch.equal(changed_good, good_h)

    # B：把第一条非空输入错误标记为全 PAD。
    bad_valid = valid.clone()
    bad_valid[0] = False
    with torch.no_grad():
        try:
            block(h, bad_valid)
        except ValueError as error:
            mask_error = str(error)
            assert '没有可读 Key' in mask_error
        else:
            raise AssertionError('预期的全屏蔽行错误未发生')

        # 这批输入本来非空，从真实 Token ID 恢复正确的 Mask。
        corrected_valid = state['input_ids'].ne(0)
        corrected_output, _ = block(h, corrected_valid)
        torch.testing.assert_close(corrected_output, state['output'], atol=1e-6, rtol=1e-5)

    assert torch.equal(h, original_h)
    result = {
        'A_alias_is_same_object': alias_same_object,
        'A_original_baseline_was_changed': baseline_corrupted,
        'A_clone_fix_preserves_baseline': True,
        'B_expected_error': mask_error,
        'B_correct_mask_fix_passed': True,
        'independent_cases_preserve_original_h': True,
    }
    print('第四题:', result)
    return result


if __name__ == '__main__':
    result = run_homework()
    debug_result = debug_exercises(result)
