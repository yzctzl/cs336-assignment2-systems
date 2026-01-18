# FlashAttention 2 前向传播等价性证明

## 标准 Attention 定义

对于查询块 $Q_i$、所有键 $K$ 和值 $V$，标准的 Attention 输出 $O_i$ 定义为：

$$O_i = \text{softmax}(Q_i K^\top) V$$

为了数值稳定性，Softmax 通常采用“减去最大值”的技巧。令 $S_i = Q_i K^\top$，全局行最大值为 $m_i^{global}$，全局指数和为 $L_i^{global}$。第 $k$ 个元素的 Softmax 计算为：

$$\text{softmax}(S_i)_k = \frac{e^{S_{i,k} - m_i^{global}}}{\sum_j e^{S_{i,j} - m_i^{global}}}$$

因此，标准输出可以写为加权和的形式：

$$O_i = \frac{\sum_{j=1}^{N} e^{S_{i,j} - m_i^{global}} V_j}{\sum_{j=1}^{N} e^{S_{i,j} - m_i^{global}}}$$

## FlashAttention-2 算法递推过程

FlashAttention-2 将 $K$ 和 $V$ 分为 $T_c$ 个块。我们在遍历第 $j$ 个块时维护三个变量：局部最大值 $m_i^{(j)}$，未归一化的指数和 $l_i^{(j)}$，以及未归一化的输出累积 $O_i^{(j)}$。
根据 Algorithm 1 (FlashAttention-2 forward pass)，第 $j$ 步的更新规则如下：

计算当前块的分数： $S_i^{(j)} = Q_i K_j^\top$。

更新最大值： $m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(S_i^{(j)}))$。

计算当前块的未归一化概率： $\tilde{P}_i^{(j)} = \exp(S_i^{(j)} - m_i^{(j)})$。

更新指数和：$ l_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} l_i^{(j-1)} + \text{rowsum}(\tilde{P}\_i^{(j)}) $

更新未归一化输出（关键改进）：
FlashAttention-2 不会在每一步进行除法归一化，而是仅通过缩放因子更新累积和：

$$O_i^{(j)} = \text{diag}(e^{m_i^{(j-1)} - m_i^{(j)}}) O_i^{(j-1)} + \tilde{P}_i^{(j)} V_j$$

## 数学归纳证明

我们需要证明，在第 $j$ 步结束时，未归一化的输出 $O_i^{(j)}$ 等于前 $j$ 个块的加权和，且所有项都已对齐到当前的最大值 $m_i^{(j)}$。

### 命题

$$O_i^{(j)} = \sum_{k=1}^{j} e^{S_i^{(k)} - m_i^{(j)}} V_k$$

### 证明

#### 基础情况 ($j=1$)：
初始化 $O_i^{(0)} = 0, m_i^{(0)} = -\infty$。
$$O_i^{(1)} = \text{diag}(e^{-\infty - m_i^{(1)}}) \cdot 0 + e^{S_i^{(1)} - m_i^{(1)}} V_1 = e^{S_i^{(1)} - m_i^{(1)}} V_1$$
命题成立。

#### 归纳步骤：
假设对于 $j-1$ 步命题成立，即：

$$O_i^{(j-1)} = \sum_{k=1}^{j-1} e^{S_i^{(k)} - m_i^{(j-1)}} V_k$$

在第 $j$ 步，根据算法的更新公式：

$$O_i^{(j)} = O_i^{(j-1)} \cdot e^{m_i^{(j-1)} - m_i^{(j)}} + e^{S_i^{(j)} - m_i^{(j)}} V_j$$

将归纳假设代入 $O_i^{(j-1)}$：

$$O_i^{(j)} = \left( \sum_{k=1}^{j-1} e^{S_i^{(k)} - m_i^{(j-1)}} V_k \right) \cdot e^{m_i^{(j-1)} - m_i^{(j)}} + e^{S_i^{(j)} - m_i^{(j)}} V_j$$

利用指数运算法则 $e^{A} \cdot e^{B} = e^{A+B}$，合并第一项的指数：

$$(S_i^{(k)} - m_i^{(j-1)}) + (m_i^{(j-1)} - m_i^{(j)}) = S_i^{(k)} - m_i^{(j)}$$

因此公式变为：

$$O_i^{(j)} = \sum_{k=1}^{j-1} e^{S_i^{(k)} - m_i^{(j)}} V_k + e^{S_i^{(j)} - m_i^{(j)}} V_j$$
合并求和项：

$$O_i^{(j)} = \sum_{k=1}^{j} e^{S_i^{(k)} - m_i^{(j)}} V_k$$

#### 归纳结论：
命题对于所有 $j$ 成立。当循环结束时（$j=T_c$），$m_i^{(T_c)}$ 即为全局最大值 $m_i^{global}$，此时：

$$O_i^{(T_c)} = \sum_{k=1}^{T_c} e^{S_i^{(k)} - m_i^{global}} V_k$$
同理可证，$l_i^{(T_c)}$ 为归一化分母：

$$l_i^{(T_c)} = \sum_{k=1}^{T_c} \text{rowsum}(e^{S_i^{(k)} - m_i^{global}})$$

#### 最终归一化：
算法的最后一步执行统一除法：

$$O_i = \text{diag}(l_i^{(T_c)})^{-1} O_i^{(T_c)} = \frac{\sum_{k=1}^{T_c} e^{S_i^{(k)} - m_i^{global}} V_k}{\sum_{k=1}^{T_c} e^{S_i^{(k)} - m_i^{global}}}$$

这正是标准 Attention 的定义公式。因此，FlashAttention-2 的前向传播算法与标准 Attention 算术等价。


# FlashAttention 2 反向传播等价性证明

FlashAttention-2 反向传播的核心思想是：利用前向传播保存的统计量（LogSumExp $L$）和输出 $O$，在片上（SRAM）重新计算注意力矩阵 $P$，从而避免存储巨大的 $N \times N$ 矩阵，并利用分块矩阵乘法计算梯度。

## 标准 Attention 的反向传播梯度公式

根据链式法则，对于损失函数 $\mathcal{L}$，标准 Attention 的梯度计算如下 1：

设 $dO = \frac{\partial \mathcal{L}}{\partial O}$ 为输出梯度。
对 $V$ 的梯度：

$$dV = P^\top dO$$
对 $P$ 的梯度（中间变量）：

$$dP = dO V^\top$$
对 $S$ 的梯度（Softmax 的反向传播）：
Softmax 的导数性质为 

$$\frac{\partial \text{softmax}(S)_j}{\partial S_k} = P_j (\delta_{jk} - P_k)$$

由此推导出 $S$ 的梯度：

$$dS_{ij} = P_{ij} \left( dP_{ij} - \sum_{k} P_{ik} dP_{ik} \right)$$
对 $Q$ 和 $K$ 的梯度：

$$dQ = dS K, \quad dK = dS^\top Q$$

## FlashAttention-2 算法的推导与证明

FlashAttention-2 将上述矩阵运算分解为块（Block）运算。假设我们将 $Q, K, V$ 分为多个块，证明过程如下：

### 第一步：重计算注意力概率 $P$
在反向传播中，算法不需要从 HBM 读取 $P$，而是根据保存的 $Q, K$ 和前向传播计算出的 LogSumExp $L$ 重新计算。

$$P_{ij} = \exp(Q_i K_j^\top - L_i)$$

由于 $L_i$ 是前向传播计算出的精确的 $\log(\sum_k \exp(S_{ik}))$，因此这里重计算出的 $P_{ij}$ 与标准前向传播中的 $P$ 完全一致。这意味着后续梯度的基础是准确的。

### 第二步：证明 $dV$ 的等价性
FlashAttention-2 算法通过遍历所有查询块 $i$ 来更新 $dV_j$：

$$dV_j \leftarrow dV_j + P_{ij}^\top dO_i$$

这对应于矩阵乘法 $P^\top dO$ 的分块形式：

$$dV = \sum_{i} P_{i,:}^\top dO_{i,:}$$

这与标准公式 $dV = P^\top dO$ 完全等价。

### 第三步：证明 $dS$ 的等价性（关键步骤）
这是最复杂的一步。标准公式中 $dS_{ij} = P_{ij} (dP_{ij} - \text{rowsum}(P \circ dP)_i)$。
FlashAttention-2 引入了一个辅助项 $D$ 来处理求和项。
预计算 $D$：
算法首先计算 $D_i = \text{rowsum}(dO_i \circ O_i)$。
我们来证明这一项等于标准公式中的 $\sum_k P_{ik} dP_{ik}$。
展开 $\sum_k P_{ik} dP_{ik}$：

$$\sum_k P_{ik} dP_{ik} = \sum_k P_{ik} (dO_i \cdot V_k^\top) = dO_i \cdot \left( \sum_k P_{ik} V_k \right)^\top$$

注意到括号中的 $\sum_k P_{ik} V_k$ 正是前向传播的输出 $O_i$。
因此：

$$\sum_k P_{ik} dP_{ik} = dO_i \cdot O_i^\top = \text{rowsum}(dO_i \circ O_i)$$

这证明了 FlashAttention-2 预先计算的 $D_i$ 正是 Softmax 梯度公式所需的修正项。
计算 $dS$：
算法在片上计算：

$$dP_{ij} = dO_i V_j^\top$$

$$dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$$

将 $D_i$ 的含义代入，得到：

$$dS_{ij} = P_{ij} (dO_i V_j^\top - \sum_k P_{ik} dP_{ik})$$

这与标准 Softmax 梯度公式完全一致。

### 第四步：证明 $dQ$ 和 $dK$ 的等价性
最后，算法利用计算出的 $dS_{ij}$ 更新 $dQ$ 和 $dK$：
对 $Q$ 的梯度：

$$dQ_i \leftarrow dQ_i + dS_{ij} K_j$$

这是矩阵乘法 $dQ = dS K$ 的分块累积形式。
对 $K$ 的梯度：

$$dK_j \leftarrow dK_j + dS_{ij}^\top Q_i$$

这是矩阵乘法 $dK = dS^\top Q$ 的分块累积形式。

### 结论
FlashAttention-2 的反向传播算法通过：重计算 精确还原了注意力矩阵 $P$；
利用 $D = dO \circ O$ 精确实现了 Softmax 的梯度修正项；
利用 分块矩阵乘法 累积了 $dQ, dK, dV$；
在数学上，上述每一步都严格遵循了标准反向传播的链式法则推导。因此，FlashAttention-2 的反向传播算法与标准 Attention 是数学等价的，且不引入任何近似误差。
