import math
import torch
from torch import nn

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)


def clear_pad(x, valid):
    return x.masked_fill(
        ~valid.unsqueeze(-1), 0.0
    )


class TinyPreNormBlock(nn.Module):
    def __init__(self, D, H, D_ff):
        super().__init__()

        if H <= 0 or D % H != 0:
            raise ValueError(
                "H 必须为正数，且 D 能被 H 整除"
            )

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

    def forward(
        self,
        x,
        valid,
        use_causal=True,
        mask_pad_keys=True,
    ):
        B, T, D = x.shape

        # 1. Pre-Norm Attention
        normed1 = self.norm1(x)

        def split_heads(tensor):
            return tensor.reshape(
                B, T, self.H, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(normed1))
        key = split_heads(self.k_proj(normed1))
        value = split_heads(self.v_proj(normed1))

        scores = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 两个开关用于对照，正常调用时均为 True。
        allowed = torch.ones(
            B, 1, T, T,
            dtype=torch.bool,
            device=x.device,
        )

        if mask_pad_keys:
            allowed = (
                allowed
                & valid[:, None, None, :]
            )

        if use_causal:
            causal = torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device,
            ).tril()

            allowed = (
                allowed
                & causal[None, None, :, :]
            )

        if not bool(allowed.any(dim=-1).all()):
            raise ValueError(
                "存在没有可读 Key 的 Query"
            )

        masked_scores = scores.masked_fill(
            ~allowed, -torch.inf
        )
        weights = torch.softmax(
            masked_scores, dim=-1
        )

        head_output = weights @ value

        merged = (
            head_output.transpose(1, 2)
            .contiguous()
            .view(B, T, D)
        )

        after_attn = clear_pad(
            x + self.out_proj(merged),
            valid,
        )

        # 2. Pre-Norm FFN
        ff_hidden = self.ff_up(
            self.norm2(after_attn)
        )

        ff_out = self.ff_down(
            self.activation(ff_hidden)
        )

        output = clear_pad(
            after_attn + ff_out,
            valid,
        )

        return output, {
            "query": query,
            "weights": weights,
            "allowed": allowed,
            "after_attn": after_attn,
            "ff_hidden": ff_hidden,
            "output": output,
        }


# 3. 输入、Embedding、Block 都只创建一次。
input_ids = torch.tensor(
    [[2, 3, 4, 0],
     [5, 4, 8, 6]],
    dtype=torch.long,
)

valid = input_ids.ne(0)
B, T = input_ids.shape

D, H, D_ff = 8, 2, 16

token_embedding = nn.Embedding(
    9, D, padding_idx=0
)
position_embedding = nn.Embedding(16, D)

block = TinyPreNormBlock(D, H, D_ff)
block.eval()


with torch.no_grad():
    # 4. 构造初始表示。
    h = token_embedding(input_ids)
    h = h + position_embedding(
        torch.arange(T)
    ).unsqueeze(0)

    h = clear_pad(h, valid)

    # 5. 正常路径检查。
    output, trace = block(h, valid)

    assert output.shape == (B, T, D)
    assert trace["ff_hidden"].shape == (B, T, D_ff)

    assert bool(torch.isfinite(output).all())
    assert bool(
        torch.isfinite(trace["weights"]).all()
    )

    assert torch.allclose(
        trace["weights"].sum(-1),
        torch.ones(B, H, T),
    )

    forbidden = ~trace["allowed"].expand_as(
        trace["weights"]
    )

    assert bool(
        (trace["weights"][forbidden] == 0).all()
    )
    assert bool((output[~valid] == 0).all())

    print("h:", tuple(h.shape))

    for name in (
        "query",
        "weights",
        "after_attn",
        "ff_hidden",
        "output",
    ):
        print(name, tuple(trace[name].shape))

    # 6. 因果实验：仅修改第二条输入的最后一个真实位置。
    changed_future = h.clone()
    changed_future[1, -1] += torch.linspace(
        -2.0, 2.0, D
    )

    causal_changed, _ = block(
        changed_future, valid
    )

    torch.testing.assert_close(
        causal_changed[1, :-1],
        output[1, :-1],
        atol=1e-6,
        rtol=1e-5,
    )

    causal_delta = (
        causal_changed[1, :-1]
        - output[1, :-1]
    ).abs().max().item()

    # 7. 对照：关闭因果限制。
    open_base, _ = block(
        h, valid,
        use_causal=False,
    )

    open_changed, _ = block(
        changed_future, valid,
        use_causal=False,
    )

    open_delta = (
        open_changed[1, :-1]
        - open_base[1, :-1]
    ).abs().max().item()

    assert not torch.allclose(
        open_changed[1, :-1],
        open_base[1, :-1],
        atol=1e-6,
        rtol=1e-5,
    )

    # 8. PAD 实验：在初始清零之后修改 PAD 表示。
    changed_pad = h.clone()
    changed_pad[~valid] = (
        torch.arange(
            1, D + 1, dtype=h.dtype
        ) * 10.0
    )

    # 为单独检验 PAD Mask，暂时关闭因果限制。
    pad_changed, _ = block(
        changed_pad, valid,
        use_causal=False,
    )

    torch.testing.assert_close(
        pad_changed[valid],
        open_base[valid],
        atol=1e-6,
        rtol=1e-5,
    )

    pad_delta = (
        pad_changed[valid]
        - open_base[valid]
    ).abs().max().item()

    # 9. 对照：进一步关闭 PAD Key 屏蔽。
    unmasked_base, _ = block(
        h, valid,
        use_causal=False,
        mask_pad_keys=False,
    )

    unmasked_changed, _ = block(
        changed_pad, valid,
        use_causal=False,
        mask_pad_keys=False,
    )

    unmasked_delta = (
        unmasked_changed[valid]
        - unmasked_base[valid]
    ).abs().max().item()

    assert not torch.allclose(
        unmasked_changed[valid],
        unmasked_base[valid],
        atol=1e-6,
        rtol=1e-5,
    )


print("因果开启，前缀最大变化:", causal_delta)
print("因果关闭，前缀最大变化:", open_delta)

print("PAD 屏蔽开启，真实位置最大变化:", pad_delta)
print("PAD 屏蔽关闭，真实位置最大变化:", unmasked_delta)

print("正常路径和两组对照实验全部通过")