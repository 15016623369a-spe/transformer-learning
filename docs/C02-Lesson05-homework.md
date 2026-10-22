# C02 Lesson05：作业答案与代码检查

更新日期：2026-10-22。

## 保存文件检查

用户已索要答案并要求检查、补齐、整理和同步。保存的 C02/lesson05.py 是完整课堂示例 D=8/H=2/D_ff=16，原有断言与对照执行通过，无需修改核心计算。

没有发现独立保存的 Lesson05 作业文件，本次新增 [lesson05exercise.py](../C02/exercise/lesson05exercise.py)，明确标记为助手参考补充。课堂原文件、私下完成确认和助手补充分别记录。

## 第一题：六个概念

1. Mask 控制信息访问，通常不改变输出形状；所以 shape 正确仍可能泄漏未来信息。
2. 同一个 Block 保持参数不变，才可将差异归因于输入或 Mask；重新创建通常会引入不同随机参数。
3. `changed=h` 是别名，原地修改会污染 h；`h.clone()` 创建独立数据副本。
4. 右侧 PAD 已被因果限制挡住；暂时关闭因果限制，才能单独检验 PAD Key Mask。
5. eval 改变特定模块的训练/评估行为，no_grad 禁用该段计算的梯度记录，两者不能替代。
6. 不能证明理解中文。本课是随机初始化、未训练模型的形状和计算规则实验，输出仍是表示向量。

## 第二题：配置与形状

```text
input_ids = [[2,3,4,0], [5,6,0,0], [2,3,7,8]]
B=3, T=4, D=12, H=3, head_dim=4, D_ff=24
```

| 对象 | shape |
|---|---|
| input_ids | [3,4] |
| h | [3,4,12] |
| query | [3,3,4,4] |
| weights | [3,3,4,4] |
| allowed | [3,1,4,4] |
| after_attn、output | [3,4,12] |
| ff_hidden | [3,4,24] |

query 末尾的 4 指每头特征数；weights 末尾的 4 指 Key 位置数，本例数值碰巧相同。

参考脚本 run_homework() 保留同一个模型，检查输出和中间形状、权重逐行和、禁止位置为零、三个 PAD 位置输出为零、正常结果有限，六组检查均通过。

## 第三题：对照实验

因果实验仅扰动第三条输入最后一个真实位置，比较该条输入前三个位置。PAD 实验在初始 clear_pad 后扰动 PAD 表示，并在两次计算中都关闭因果限制，比较所有真实位置。

| 实验 | 本次最大绝对变化 |
|---|---:|
| 因果开启，前缀 | 0 |
| 因果关闭，前缀 | 0.0823882222 |
| PAD 屏蔽开启，真实位置 | 0 |
| PAD 屏蔽关闭，真实位置 | 0.1807783842 |

受保护位置保持一致，关闭规则后的对照能产生变化。关闭限制只允许影响，不保证任何参数和扰动都必然产生明显差异；因此需要核对对照有效性，不死记小数。

## 第四题 A：别名污染基线

```python
changed = h
changed[2, -1] += torch.linspace(-2.0, 2.0, D)
```

通常不报错，但 changed 与 h 为同一个张量，基线已改变。参考脚本使用局部 bad_h 演示，避免污染其他实验。

```python
before = h.clone()
changed = h.clone()
changed[2, -1] += torch.linspace(-2.0, 2.0, D)
assert torch.equal(h, before)
assert not torch.equal(changed, h)
```

若先前已运行错误写法，应先用原有 Embedding 参数重建干净 h；clone 不能找回已覆盖的数据。

## 第四题 B：全屏蔽行

```python
bad_valid = valid.clone()
bad_valid[0] = False
block(h, bad_valid)
```

预期报错：`ValueError: 存在没有可读 Key 的 Query`。这使第一条输入所有 Key 都不可读；直接对全为 -inf 的一行做手工 Softmax 会出现 NaN。

这批输入本来非空，应从实际编号恢复正确 Mask：

```python
corrected_valid = input_ids.ne(0)
corrected_output, _ = block(h, corrected_valid)
```

真实空输入或全 PAD 样本应在进入本课 Block 前拒绝或过滤。本例约定每条输入至少有一个真实 Token，采用右侧补齐；保留 Softmax 前的全屏蔽行保护。

## 验证入口

```bash
python C02/lesson05.py
python C02/exercise/lesson05exercise.py
python C02/verify.py --report /tmp/c02-verification.json
```

课堂、作业两组均与同参数 PyTorch 原生 Pre-Norm 层对照一致。详情见 [本课记录](C02-Lesson05-verification.json) 与 [模块验收](C02-acceptance.md)。
