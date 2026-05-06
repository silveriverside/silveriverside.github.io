---
title: DEQ
tags: ["控制论与动力学"]
---

# DEQ - 深度等方程 (Deep Equilibration)

## 概述

DEQ（Deep Equilibration）是一种训练深度神经网络的方法，通过求解隐式方程来定义网络输出，而不是使用传统的逐层前向传播。

## 核心思想

传统网络：
```
y = f_L(f_{L-1}(...f_1(x)...))
```

DEQ 网络：
```
y = f(y, x)  其中 y 是均衡点，满足 y = f(y, x)
```

通过不动点迭代求解：
```python
def forward_deq(f, x, max_iter=100, tol=1e-4):
    y = torch.zeros_like(x)
    for _ in range(max_iter):
        y_new = f(y, x)
        if torch.norm(y_new - y) < tol:
            break
        y = y_new
    return y
```

## 优势

1. **内存效率**：无需存储中间层激活
2. **可扩展深度**：理论上支持无限深度
3. **隐式正则化**：均衡点有自然平滑特性

## 与传统网络的对比

| 特性 | 传统网络 | DEQ |
|-----|---------|-----|
| 深度 | 有限 | 无限（隐式） |
| 内存 | O(L) | O(1) |
| 梯度 | 反向传播 | 隐式微分 |

## 应用

- **DEQ-Seg**：图像分割
- **DEQ-NER**：命名实体识别
- **Transformer**：可视为一种 DEQ

## 关联笔记

- [隐式深度学习](./隐式深度学习/README.md)
- [Neural ODEs](./NeuralODEs/README.md)
- [状态空间模型](./状态空间模型/README.md)

---

*更新时间: 2026-04-03*

## Related Notes

[[隐式深度学习]] · [[NeuralODEs]] · [[控制论]] · [[吸引子]]
