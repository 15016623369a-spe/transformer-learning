# C02 Lesson05：综合实验与验收

更新日期：2026-10-22。课堂入口为 [lesson05.py](../C02/lesson05.py)，本次保留原内容；[作业脚本](../C02/exercise/lesson05exercise.py) 是明确标记的助手参考补充。

## 五个重点

1. **形状、数值与行为分开检查。** 输出 shape 正确不能证明没有未来信息泄漏。
2. **比较时固定参数。** 同一个 Block 多次前向，只改变本次研究的输入或 Mask。
3. **使用 clone 保留基线。** 普通赋值只是另起变量名，原地修改会影响同一个张量。
4. **因果实验比较前缀；PAD 实验比较真实位置。** 两类实验关注的信息通路不同。
5. **检查右侧 PAD 时暂时关闭因果限制。** 扰动放在初始清零之后，并用关闭 PAD Key 屏蔽的对照确认实验有区分度。

## 1. 三层检查

| 层次 | 本课检查 |
|---|---|
| 形状 | h/output 为 [B,T,D]，query 为 [B,H,T,D/H]，weights 为 [B,H,T,T]，FFN 中间为 [B,T,D_ff] |
| 数值 | 正常输出与权重有限；权重沿 Key 逐行和约为 1；禁止位置权重为 0；PAD 最终输出为 0 |
| 行为 | 改变未来位置不影响前缀；改变 PAD 表示不影响真实位置；关闭相应屏蔽后，对照可产生变化 |

只有负无穷屏蔽分数和明确标注的错误实验可以包含非有限值，不能把它们混入正常输出检查。

## 2. 因果实验

```text
固定模型、其余输入、位置与 Mask
仅修改最后一个真实位置
比较它之前的位置输出
```

开启因果限制时，前缀不能读取未来位置，因而应保持一致。关闭限制只表示允许受到影响；不是所有参数或扰动都必然产生明显变化，所以本课还核对了对照的差异。

这里直接扰动 h，是为了检验计算依赖，不是执行真实分词或证明语言理解。不要将同一 Token 全部特征仅加一个常数作为唯一扰动：LayerNorm 会消去这种整体平移，可能让实验失去区分度。

## 3. PAD 实验为何需要单独设计

右侧 PAD 位于真实 Token 之后，因果限制本来就会挡住它；因此单独检查 Padding Mask 时设置 use_causal=False。

此外，如果改完 PAD 表示又执行初始 clear_pad，扰动会被清除。本课在初始清零后修改送入 Block 的 PAD 表示，再分别开启、关闭 PAD Key 屏蔽。

**屏蔽 PAD Key 与清零 PAD 输出是两个动作。** 即使关闭 mask_pad_keys，示例仍会在残差后清零 PAD 输出；我们检查的是真实位置是否受 PAD 内容影响。

## 4. 参数复用、模式与比较

- `__init__` 创建层；`forward` 使用已有层。`block(h, valid)` 复用同一个模块及其参数。
- `changed = h` 指向同一张量；`changed = h.clone()` 创建独立数据副本。
- eval 切换模块的评估模式，不自动关闭梯度记录；no_grad 控制这段前向计算的梯度记录。
- assert_close 使用容差比较浮点结果；本课采用 atol=1e-6、rtol=1e-5。最大绝对差由 `(a-b).abs().max().item()` 得到。

对应 API 说明：[Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)、[no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html)、[assert_close](https://docs.pytorch.org/docs/2.14/testing.html#torch.testing.assert_close)。

## 5. 完整教师代码

```python
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
```

## 6. 实际结果

| 实验 | 课堂 D=8/H=2 | 作业 D=12/H=3 |
|---|---:|---:|
| 因果开启，前缀最大变化 | 0 | 0 |
| 因果关闭，前缀最大变化 | 0.087648 | 0.082388 |
| PAD 屏蔽开启，真实位置最大变化 | 0 | 0 |
| PAD 屏蔽关闭，真实位置最大变化 | 0.142412 | 0.180778 |

以上是本次种子与层创建顺序下的 CPU 结果；不要求其他实现逐项复现非零小数。两组正常输出均与复制相同参数的原生 Pre-Norm 层一致。

## 7. 易错点

| 问题 | 修正方向 |
|---|---|
| 两次比较重新创建模型 | 固定同一个 Block |
| 把 changed=h 当成独立副本 | 使用 clone，并恢复已污染的基线 |
| 修改未来位置后比较整个输出 | 只比较受因果规则保护的前缀 |
| 输入清零把 PAD 扰动抹掉 | 在初始清零之后扰动 |
| 因果限制掩盖 PAD Key Mask 错误 | PAD 对照暂时关闭因果限制 |
| 仅检查“不变”，不检查扰动有效性 | 增加关闭对应屏蔽的对照 |
| 为通过检查随意扩大容差 | 先定位计算顺序、参数、位置与 Mask |
| 全 PAD 样本直接进入手工 Softmax | 在输入阶段处理，保留全屏蔽行保护 |

## 8. 作业与复核

四题包括：六个概念解释；B=3/T=4/D=12/H=3/D_ff=24 正常路径；因果与 PAD 两组对照；别名与全屏蔽行两个问题案例。完整答案见 [作业与检查](C02-Lesson05-homework.md)。

本课只做未训练模型的前向计算，不能由测试通过推断语言理解或用户独立掌握。用户已索要答案，按既定约定接受为私下完成确认。
