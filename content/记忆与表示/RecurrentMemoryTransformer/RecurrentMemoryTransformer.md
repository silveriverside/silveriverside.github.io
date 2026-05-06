---
title: RecurrentMemoryTransformer
tags: ["记忆与表示"]
---

# Recurrent Memory Transformer (RMT)

## 概述

RMT 在 Transformer 中引入记忆机制，通过循环传递隐藏状态来处理长序列。

## 核心思想

### 全局记忆
- 添加记忆 token (memory tokens)
- 记忆在序列间循环传递

```python
class RMT(nn.Module):
    def __init__(self, d_model, n_memory):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_memory, d_model))

    def forward(self, x, memory):
        # 将记忆与输入拼接
        x = torch.cat([memory, x], dim=0)
        x = self.transformer(x)
        # 分离输出和更新后的记忆
        new_memory = x[:len(memory)]
        output = x[len(memory):]
        return output, new_memory
```

### 分段处理
```
序列 → 分段 → 每段处理 → 记忆传递 → 下一段
```

## 与传统 Transformer 的区别

| 特性 | 标准 Transformer | RMT |
|-----|-----------------|-----|
| 序列长度 | 有限 (平方) | 可无限 |
| 记忆 | 无 | 有 (循环) |
| 计算 | 全量 | 分段 |

## 变体

1. **RMT-CL**：用于连续学习
2. **RMT-NTM**：结合神经图灵机

## 关联笔记

- [Universal Transformer](../../Transformer变体/UniversalTransformer/README.md)
- [吸引子](./吸引子/README.md)
- [状态空间模型](../../控制论与动力学/状态空间模型/README.md)

---

*更新时间: 2026-04-03*

## Related Notes

[[权重共享递归结构]] · [[UniversalTransformer]] · [[吸引子]]
