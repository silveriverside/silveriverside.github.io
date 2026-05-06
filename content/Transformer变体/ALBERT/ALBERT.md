---
title: ALBERT
tags: ["Transformer变体"]
---

# ALBERT - A Lite BERT

## 概述

ALBERT 是 Google 提出的轻量级 BERT，通过参数共享和分解Embedding来大幅减少参数量。

## 核心创新

### 1. 跨层参数共享
```
ALBERT 所有 Transformer 层共享同一组参数
```
- 减少参数量 12x
- 轻微影响性能

### 2. 词向量分解
```
原始: Embedding(V, H)
分解后: Embedding(V, E) × Linear(E, H)
```
- V: 词表大小
- H: 隐藏层维度
- E: 较小的Embedding维度

### 3. 句子顺序预测 (SOP)
替换 NSP 为 SOP：
- 正样本：连续句子
- 负样本：交换顺序的连续句子

## 参数量对比

| 模型 | 参数量 | 层数 | 隐藏维度 |
|-----|-------|-----|---------|
| BERT-Large | 340M | 24 | 1024 |
| ALBERT-Large | 18M | 24 | 1024 |
| ALBERT-xxlarge | 12M | 12 | 2048 |

## 性能

- GLUE 分数接近 BERT-Large
- 推理速度更快
- 内存需求更低

## 关联笔记

- [Universal Transformer](./UniversalTransformer/README.md)
- [权重共享递归结构](../../记忆与表示/权重共享递归结构/README.md)

---

*更新时间: 2026-04-03*

## Related Notes

[[UniversalTransformer]] · [[权重共享递归结构]]
