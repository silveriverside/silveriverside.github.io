---
title: NeuralODEs
tags: ["控制论与动力学"]
---

# Neural ODEs - 神经网络常微分方程

## 概述

Neural ODEs 将神经网络与连续时间动力学系统结合，用微分方程来定义网络层。

## 核心方程

```
dy/dt = f(y, t, θ)
```

其中 f 是由神经网络参数化的函数。

## 求解方法

### 龙格-库塔法 (Runge-Kutta)
```python
def rk4_step(f, y, t, dt, theta):
    k1 = f(y, t, theta)
    k2 = f(y + dt/2*k1, t + dt/2, theta)
    k3 = f(y + dt/2*k2, t + dt/2, theta)
    k4 = f(y + dt*k3, t + dt, theta)
    return y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
```

### 自适应步长方法
- dopri5 (Dormand-Prince)
- adams

## 反向传播：Adjoint 方法

使用伴随方法计算梯度，避免存储中间状态：
```python
def adjoint_loss(y1, yT):
    return torch.norm(y1 - yT)**2

# 自动微分计算伴随梯度
grad = torch.autograd.grad(adjoint_loss, theta)
```

## 应用场景

1. **时间序列建模**：连续时间 RNN
2. **密度估计**：FFJORD
3. **图像生成**：Neural ODE 流模型

## 与传统网络的比较

| 特性 | 标准 ResNet | Neural ODE |
|-----|------------|------------|
| 时间 | 离散 | 连续 |
| 梯度 | BPTT | 伴随方法 |
| 表示能力 | 有限步 | 无限分辨率 |

## 关联笔记

- [控制论原理](../控制论/README.md)
- [DEQ](./DEQ/README.md)
- [状态空间模型](./状态空间模型/README.md)

---

*更新时间: 2026-04-03*

## Related Notes

[[DEQ]] · [[控制论]] · [[吸引子]] · [[状态空间模型]]
