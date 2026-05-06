---
title: UniversalTransformer
tags: ["Transformer变体"]
---

# Universal Transformer

## 概述

Universal Transformer 是 Google 提出的 Transformer 变体，结合了 Transformer 的并行计算能力和 RNN 的循环特性。

## 核心创新

### 1. 循环机制
不是固定的 N 层，而是循环 T 次：
```python
def universal_transformer(x, T):
    h = embedding(x)
    for t in range(T):
        h = transformer_layer(h)
        h = adaptive_embedding(h)  # 位置编码可学习
    return output(h)
```

### 2. 动态停止
对不同位置使用不同数量的计算步骤：
```python
def conditional_stopping(h, max_steps):
    for t in range(max_steps):
        h = transformer_layer(h)
        if should_stop(h):
            break
    return h
```

### 3. 自回归解码
使用循环机制进行自回归生成：
```python
def decode(enc_output, max_len):
    for i in range(max_len):
        output = transformer_decoder(output, enc_output)
        if output == EOS:
            break
```

## 与标准 Transformer 对比

| 特性 | Transformer | Universal Transformer |
|-----|------------|---------------------|
| 层数 | 固定 N | 动态 T |
| 位置编码 | 固定 | 可学习 + 循环 |
| 计算 | 并行 | 循环 + 并行 |

## 优势

1. **计算效率**：对简单模式用更少计算
2. **泛化能力**：循环结构更通用
3. **理论深度**：可表示任意深度

## 关联笔记

- [ALBERT](./ALBERT/README.md)
- [权重共享递归结构](../../记忆与表示/权重共享递归结构/README.md)
- [Recurrent Memory Transformer](../../记忆与表示/RecurrentMemoryTransformer/README.md)

---

*更新时间: 2026-04-03*

## Related Notes

[[ALBERT]] · [[权重共享递归结构]] · [[隐式深度学习]]
