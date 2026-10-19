# C02 Lesson01：作业对照与第四题解释

日期：2026-10-19（Asia/Shanghai）。用户已确认作业完成。以下为课堂参考答案与整理说明，不把助手补充的单元格当作用户独立实现的证据。

完整可运行代码见 [Notebook](../C02/lesson01.ipynb)。

## 第一题：概念

1. Token 是切分后的单位；ID 是词表索引；Embedding 是查出的向量。
2. ID 大小与编号距离没有直接语义意义，不能用 ID 的差值判断相似性。
3. V 是词表大小，T 是当前序列长度；本课 T 包含补齐位置。
4. 相同 ID 在同一参数表中查出的向量相同，后续表示可因位置和上下文计算而不同。
5. 随机参数没有学到可用语义；Token 级表示也不等于文本级检索表示，不能直接作为可用 RAG 模型。

## 第二题：shape

给定 B=3、T=5、V=12、D=6：

| 对象 | shape | 理由 |
|---|---|---|
| input_ids | [3,5] | 三条输入、每条五个位置 |
| embedding.weight | [12,6] | 十二行、每行六个数 |
| x | [3,5,6] | 每个输入位置对应六维向量 |
| valid_mask | [3,5] | 每个输入位置对应一个标记 |
| x[1] | [5,6] | 第二条输入的全部位置 |
| x[1,3] | [6] | 第二条输入第四个位置的向量 |

参数数值总量为 `12×6=72`。下标从零开始。

## 第三题：扩展词表

原词表末尾增加“退款”及“分析”，编号分别为 7、8。B=3、T=4、V=9、D=6。

课堂题目顺序为“我 查询 订单”“查询 退款”“生成 订单 分析 报告”；用户 Notebook 将后两条交换。本次保留用户顺序，因此编号及 mask 的行顺序与课堂参考答案相应不同，shape 与查表原理不变。

```text
input_ids:        [3,4]
embedding.weight: [9,6]
x:                [3,4,6]
```

“查询”出现在 `x[0,1]` 与 `x[2,0]`，两处查表向量相同。

单独处理“我 查询 库存”时，词表未收录“库存”，编号为 `[[2,3,1]]`，有效标记为 `[[1,1,1]]`。UNK 是未知内容，不是补齐内容。单独一条输入无需为了对齐另一批次而补到 T=4。

## 第四题：类型错误与索引越界

`nn.Embedding(7,4)` 有七行、每行四个数。合法 ID 为 0～6。

### 案例 A：浮点编号

```python
layer = nn.Embedding(7, 4)
ids = torch.tensor([[2.0, 3.0]])
result = layer(ids)
```

原题默认配置下，`ids.dtype` 为 `torch.float32`。Embedding 支持整数索引，不接受浮点索引。报错要求 Long 或 Int，但实际得到 FloatTensor。

修正：

```python
ids = torch.tensor([[2, 3]], dtype=torch.long)
result = layer(ids)
# result.shape == [1,2,4]
```

编号 2 和 3 在范围内，修复的是类型。`.long()` 也能转换本题恰好为整数的数值，但不能把截断任意小数当成正确的词表映射。

### 案例 B：编号超出表范围

```python
layer = nn.Embedding(7, 4)
ids = torch.tensor([[7]], dtype=torch.long)
result = layer(ids)
```

类型正确，但 ID=7 要访问第八行，当前表只有七行。实际报错为 `IndexError: index out of range in self`。

原题没有说明 7 原本要表示什么，不能唯一恢复意图。参考修正假设输入的是“报告”，按原七项词表查得 ID=6：

```python
original_vocab = ["<PAD>", "<UNK>", "我", "查询", "订单", "生成", "报告"]
original_mapping = {token: idx for idx, token in enumerate(original_vocab)}
layer = nn.Embedding(len(original_vocab), 4)
ids = torch.tensor([[original_mapping["报告"]]], dtype=torch.long)
result = layer(ids)
# result.shape == [1,1,4]
```

不能随意挑一个合法数字，也不能仅为消除报错扩大层的行数。新 Token 需要词表与参数表一致变更；未知 Token 在本课规则下映射为 UNK。

第三题的九行表中，ID=7 对应“退款”，是合法的；第四题重新创建的七行表中，ID=7 越界。合法范围由当前表决定。

Notebook 中有意触发的两个错误使用 `try/except` 分别展示，预期错误被捕获后仍继续执行修正代码。整本成功执行不代表原错误输入被模型接受。
