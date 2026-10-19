# C02 Lesson01：Token、Token ID、Embedding 与模型输入

授课日期：2026-10-18。作业对照与整理日期：2026-10-19（Asia/Shanghai）。

本课已完成：用户确认作业完成；本次整理对 Notebook 从干净内核执行验证。可运行实践见 [lesson01.ipynb](../C02/lesson01.ipynb)。

## 1. 从文字到向量

```text
原始文字 → Tokenizer 切分 → Token → Token ID → Embedding 查表 → 各位置的向量
```

- Token：切分后的单位，不保证等于一个中文字符或一个英文单词。
- Tokenizer：负责切分及编号；真实模型需要使用配套的 Tokenizer。
- Token ID：词表中的整数索引，不是语义分数。
- Token Embedding：按 ID 从参数表取出的向量。Embedding 也可以指执行这项操作的层。

本课使用带空格的教学文本及 `text.split()`。`"我 查询 订单".split()` 得到三个 Token，`"我查询订单".split()` 则只得到一个元素。这不是通用中文分词器，也不能据此估计真实模型的 Token 数。

编号之间的距离没有直接语义含义。把 ID 转成浮点数也不等于生成 Embedding。

## 2. B、T、V、D 与 shape

| 符号 | 含义 |
|---|---|
| B | 一次处理的输入条数 |
| T | 当前批次补齐后的序列长度 |
| V | 词表的 Token 数量，也是本课 Embedding 表的行数 |
| D | 每个 Token 向量的维度 |

```text
input_ids:        [B,T]   整数编号
embedding.weight: [V,D]   浮点参数表
x:                [B,T,D] 查表结果
valid_mask:       [B,T]   有效位置标记
```

查表关系：`x[b,t] = embedding.weight[input_ids[b,t]]`。每个输入位置原来放一个 ID，输出中同一位置放一个 D 维向量。Embedding 不会自动把整句话压缩成一个向量。

单条输入 `[T]` 经查表输出 `[T,D]`；显式保留批量维度 `[1,T]` 则输出 `[1,T,D]`。本课统一采用批量形式。

`embedding` 是层对象；`embedding.weight` 是整张参数表；`x = embedding(input_ids)` 是本次调用的结果。新输入通常复用同一层，不必重新初始化。

## 3. PAD、UNK 与有效位置

`<PAD>` 用于补齐，原文中没有对应内容；`<UNK>` 表示原文有内容，但本课词表未收录该 Token。

本课的编号：

| ID | Token |
|---:|---|
| 0 | `<PAD>` |
| 1 | `<UNK>` |
| 2 | 我 |
| 3 | 查询 |
| 4 | 订单 |
| 5 | 生成 |
| 6 | 报告 |
| 7 | 退款 |
| 8 | 分析 |

`valid_mask = input_ids.ne(pad_id)`，其中 `True` 为有效位置，`False` 为补齐位置。UNK 仍然是有效内容，不能当成 PAD。

新建 `nn.Embedding(..., padding_idx=pad_id)` 时，PAD 行默认是零，经查表产生的梯度不会更新该行。零向量与 Attention 掩码作用不同；后续计算仍需要正确使用掩码。本课只构造标记，不包含 Attention。

检查 PAD 时按 `x[~valid_mask]` 提取补齐位置，不硬编码某个位置。输入顺序或长度改变后，固定下标可能指向普通 Token。

## 4. 随机参数、训练与上下文

`nn.Embedding(V,D)` 创建初始化参数，不会下载预训练模型，也不会自动训练。随机向量之间的相似性不能作为理解语义的证据。

训练需要损失、反向传播与优化器更新。本课没有训练这些参数，只验证查表行为。

同一张表在参数未改变时，同一个 ID 查出的向量相同。后续 Transformer 结合上下文计算，才可能使同一个 Token 的后续表示不同。

查表保留输入的排列顺序，但不会自动将位置编号编码进每个向量。位置机制留到 Lesson02。

## 5. 与 RAG 向量的区别

本课 Token 向量对应每个位置，输出 `[B,T,D]`。常见单向量 RAG 检索表示对应查询或文本片段，批量形式可写成 `[B,D_r]`。表示单位、训练目标与生成方式不同，不能把本课随机 Token 向量简单平均后称为已经可用的检索模型。

## 6. 本次实践与易错点

保留用户的三条输入顺序，实际编号为：

```text
[[2,3,4,0],
 [5,4,8,6],
 [3,7,0,0]]
```

有效位置标记为：

```text
[[1,1,1,0],
 [1,1,1,1],
 [1,1,0,0]]
```

本次整理修正了固定 PAD 下标，并清除了与当前源代码不一致的旧输出，再从头执行。

容易混淆的点：V 与 T、ID 与向量、整张表与当前输出、随机初始化与训练、有效位置标记与实际使用掩码，以及 Notebook 的当前源代码与历史输出。

## 官方参考

- [PyTorch nn.Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
- [Hugging Face：Tokenization algorithms](https://huggingface.co/docs/transformers/main/tokenizer_summary)
