# C02 Lesson04：作业答案与文件检查

整理日期：2026-10-22（Asia/Shanghai）。用户已索要 Lesson04 答案，按授课约定接受为私下完成确认；本次明确要求检查文件并同步 GitHub。

**检查结论：已保存的 C02/lesson04.py 为课堂配置 D=8、H=2、D_ff=16，原有 7 项检查通过，未发现本例核心计算错误，文件原样保留。** 目录中未发现独立的 Lesson04 作业脚本或 Notebook；本次新增 `C02/exercise/lesson04exercise.py`，明确标记为助手参考补充，不将其视为用户独立实现的证据。

## 第一题：六个概念答案

| 问题 | 参考答案 |
|---|---|
| Attention 与 FFN 谁跨位置交流？ | Attention 汇集允许读取的 Token 的信息；FFN 逐位置加工，所有位置共享同一套参数。FFN 输入可以已包含上下文。 |
| D 与 D_ff 有什么区别？ | D 为主干表示维度；D_ff 为 FFN 中间宽度。增大 D_ff 不改变 Token 数 T，也不改变最终输出 D。 |
| 为什么需要激活函数？ | 没有非线性时，两次仿射变换仍可合并为一次。本课使用 GELU。 |
| FFN 为什么要返回 D？ | 与 after_attn 做残差相加时，需要回到本题的 [B,T,D]。 |
| LayerNorm(D) 的统计范围？ | 固定样本和 Token，仅在最后 D 个特征上计算均值、方差，方差分母为 D；gamma/beta 按特征学习并跨位置共享。 |
| Pre-Norm 与 Post-Norm？ | Pre：u=x+Attention(LN1(x))；y=u+FFN(LN2(u))。Post：u=LN1(x+Attention(x))；y=LN2(u+FFN(u))。 |

LayerNorm 的方差与末尾维规则以 [PyTorch 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html) 为准。上述公式省略 Dropout 与 PAD 清零；实际脚本保留两次 PAD 清零。

## 第二题：手算 LayerNorm

给定 x=[0,2,2,4]，gamma=1、beta=0、eps=1e-5。

```text
均值 = (0+2+2+4)/4 = 2
方差 = ((0-2)^2+(2-2)^2+(2-2)^2+(4-2)^2)/4 = 2
输出 = [-2,0,0,2] / sqrt(2+1e-5)
     ≈ [-1.414210,0,0,1.414210]
```

```python
x = torch.tensor([0., 2., 2., 4.])
mean = x.mean()
variance = x.var(correction=0)
manual = (x - mean) / torch.sqrt(variance + 1e-5)
actual = nn.LayerNorm(4, elementwise_affine=False)(x)
assert torch.allclose(manual, actual, atol=1e-6)
```

参考脚本的 layernorm_exercise() 执行这项计算，实际结果与手算公式一致。课堂文件末尾原有 `[1,2,3,4]` 小例子用于教学观察；它与本题给定输入不同，两者均保留。

## 第三题：Pre-Norm Block 与 FFN 宽度比较

三条输入“我 查询 订单”“生成 报告”“我 查询 退款”。右侧补齐后的 IDs：

```text
[[2,3,4],
 [5,6,0],
 [2,3,7]]
```

B=3、T=3、D=12、H=3，每头维度 4。完整代码在参考脚本的 run_experiment(D_ff) 中，直接运行文件会依次执行 D_ff=24 和 48。

```python
normed1 = norm1(h)
# 用 normed1 生成 Q/K/V，完成带因果与 Padding Mask 的多头 Attention。
after_attn = clear_pad(h + attn_out)
normed2 = norm2(after_attn)
ff_hidden = ff_up(normed2)
ff_active = activation(ff_hidden)
ff_out = ff_down(ff_active)
output = clear_pad(after_attn + ff_out)
```

| 张量 | D_ff=24 | D_ff=48 |
|---|---|---|
| h、normed1 | [3,3,12] | [3,3,12] |
| query、key、value | [3,3,3,4] | [3,3,3,4] |
| scores、weights | [3,3,3,3] | [3,3,3,3] |
| allowed | [3,1,3,3] | [3,1,3,3] |
| attn_out、after_attn、normed2 | [3,3,12] | [3,3,12] |
| ff_hidden、ff_active | [3,3,24] | [3,3,48] |
| ff_out、output | [3,3,12] | [3,3,12] |

**只有 FFN 中间宽度随 D_ff 改变，最终形状不变；输出数值不要求相同。** 本脚本两次重置相同随机种子，并在 FFN 前构造相同输入与 Attention 层，本次也实际核对了 h 与 after_attn 相同。

每组均检查：残差各张量形状一致、中间宽度正确、最终输出形状正确、权重每行和约为 1、禁止位置权重为 0、PAD 最终输出为 0、正常计算张量无 NaN/Inf。全屏蔽行会先触发 ValueError，不会进入 Softmax。

## 第四题：错误原因与修正

使用课堂维度 B=2、T=4、D=8、D_ff=16；两个案例由 debug_exercises() 分别触发并捕获，确保后续修正仍能执行。

### A：after_attn + ff_hidden

```text
after_attn: [2,4,8]
ff_hidden:  [2,4,16]
核心报错：The size of tensor a (8) must match the size of tensor b (16)
```

错误在于把 FFN 中间表示用作残差分支输出。应完成激活和降维：

```python
ff_out = ff_down(activation(ff_hidden))  # [2,4,8]
output = after_attn + ff_out
```

带 PAD 的完整 Block 还应对 output 清零 PAD。不要通过随意裁剪、补零或改主干宽度来掩盖错误。

### B：nn.LayerNorm(T)(h)

```text
h: [2,4,8]，T=4，D=8
核心报错：Given normalized_shape=[4], expected input with shape [*, 4],
         but got input of size [2, 4, 8]
```

LayerNorm(4) 检查末尾维是否为 4；h 的末尾维是 8。按本题“逐 Token 归一化特征”的目标，应改为：

```python
result = nn.LayerNorm(D)(h)  # [2,4,8]
```

不要因为代码能运行，就把归一化范围改成整个 T×D；那会改变计算含义。

## 真实验证

- 原始课堂脚本在编辑前独立执行通过，原有 7 项检查全为 True，文件字节保持不变。
- 作业参考两种宽度各 7 项检查通过，LayerNorm 手算一致，两个预期错误均复现且修正通过。
- 课堂 D=8/H=2/D_ff=16、参考 D=12/H=3/D_ff=24 和 48，三组与同参数 PyTorch Pre-Norm TransformerEncoderLayer 对照一致；双方按相同规则清零 PAD。
- 额外检查课堂分支零输出时残差保留输入、修改未来位置不影响前缀、FFN 逐位置处理、全屏蔽行被拒绝。
- fde、Python 3.11.16、PyTorch 2.14.0、CPU；无训练、依赖变更或付费 API 调用。

机器可读结果见 [验证记录](C02-Lesson04-verification.json)。本次文件复核完成不等于整个 C02 的实验验收已经完成。
