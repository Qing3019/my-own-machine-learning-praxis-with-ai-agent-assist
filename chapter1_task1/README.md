# Project 01：数据预处理小实验

> 对应课本：第 1 章《监督学习与回归》—— 数据预处理技术

---

## 1. 项目目标

理解**为什么机器学习不能直接使用原始数据**，并掌握 5 种核心数据预处理方法。

---

## 2. 知识点总结

### 2.1 为什么需要数据预处理？

原始数据通常有两类问题：

| 问题 | 例子 | 后果 |
|------|------|------|
| 数值范围不一致 | 面积 50~200，收入 5000~50000 | 模型被大数值"带偏" |
| 文字标签无法计算 | 城市名"北京""上海" | 模型只认数字，不认文字 |

### 2.2 五种预处理方法一览

| 方法 | 操作方向 | 公式（简化） | 结果特点 | 适用场景 |
|------|---------|-------------|---------|---------|
| **StandardScaler** 标准化 | 按列 | (x - 均值) / 标准差 | 均值≈0, 标准差≈1 | SVM、逻辑回归、神经网络 |
| **MinMaxScaler** 范围缩放 | 按列 | (x - min) / (max - min) | 所有值 ∈ [0, 1] | 图像像素、神经网络输入层 |
| **normalize** 归一化 | **按行** | x / 向量长度 | 每行向量长度 = 1 | 文本分析（TF-IDF 后）、余弦相似度 |
| **LabelEncoder** 标签编码 | 整列 | 类别 → 0, 1, 2... | 一列整数 | 有顺序的类别（低/中/高） |
| **OneHotEncoder** 独热编码 | 整列 | 每类别一列 0/1 | N 列 0/1 矩阵 | **无**顺序的类别（颜色、城市） |

### 2.3 核心区别：标签编码 vs 独热编码

```
      原始: ['cat', 'dog', 'fish', 'dog', 'cat']

标签编码:  [0, 1, 2, 1, 0]
           ↑ 模型会以为 fish(2) > dog(1) > cat(0)  ← 虚假的大小关系！

独热编码:  [[1, 0, 0],  ← cat
            [0, 1, 0],  ← dog
            [0, 0, 1],  ← fish
            [0, 1, 0],  ← dog
            [1, 0, 0]]  ← cat
           ↑ 只有 0 和 1，没有大小关系 ← 正确！
```

**规则**：类别没有天然顺序 → 用独热编码；类别有天然顺序（小/中/大）→ 用标签编码。

### 2.4 核心区别：按列 vs 按行

```
StandardScaler / MinMaxScaler：对每一列（特征）独立操作
    ┌─────────┐
    │ 列1  列2 │   列1 缩放自己的范围
    │ 50   300 │   列2 缩放自己的范围
    │ 200 1200 │   → 两列互不影响
    └─────────┘

normalize：对每一行（样本）独立操作
    ┌─────────┐
    │ 50   300 │   这一行作为一个向量 → 缩放到长度 = 1
    │ 200 1200 │   每一行各自缩放
    └─────────┘
```

---

## 3. 代码函数详解

### 3.1 `StandardScaler` — 标准化

```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
result = std_scaler.fit_transform(data)
```

| 步骤 | 做了什么 |
|------|---------|
| `fit` | 计算每一列的**均值**和**标准差** |
| `transform` | 对每个值执行 `(x - 均值) / 标准差` |
| `fit_transform` | 上面两步一起做 |

- **参数**：通常不需要传参数，使用默认即可
- **输入**：二维数组，shape 为 `(样本数, 特征数)`
- **输出**：同 shape 的二维数组，每列均值 ≈ 0，标准差 ≈ 1

---

### 3.2 `MinMaxScaler` — 范围缩放

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
result = minmax_scaler.fit_transform(data)
```

| 步骤 | 做了什么 |
|------|---------|
| `fit` | 计算每一列的**最小值**和**最大值** |
| `transform` | 对每个值执行 `(x - min) / (max - min)` |
| `fit_transform` | 上面两步一起做 |

- **参数**：`feature_range=(0, 1)`（默认），可以改成 `(0, 2)` 等
- **输入**：二维数组
- **输出**：值全部落在 `feature_range` 范围内

---

### 3.3 `normalize` — 归一化（注意：是函数，不是类！）

```python
from sklearn.preprocessing import normalize

result = normalize(data)  # 直接调用函数，不需要先创建实例
```

| 参数 | 含义 |
|------|------|
| `data` | 二维数组 |
| `norm` | `'l2'`（默认，欧氏距离长度=1）、`'l1'`（绝对值之和=1）、`'max'`（最大值=1） |

- **默认 L2 范数**：$$\sqrt{x_1^2 + x_2^2 + ... + x_n^2} = 1$$
- 和 `StandardScaler` 最大的区别：**按行操作**，不是按列

---

### 3.4 `LabelEncoder` — 标签编码

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
result = label_encoder.fit_transform(labels)  # 一维列表，不需要 reshape
```

| 步骤 | 做了什么 |
|------|---------|
| `fit` | 找出所有唯一类别，按字母顺序排序：`['cat', 'dog', 'fish']` → `{cat:0, dog:1, fish:2}` |
| `transform` | 把每个类别替换成对应整数 |
| `inverse_transform` | 把整数还原成类别（反向操作） |

- **输入**：一维列表或数组
- **输出**：一维整数数组
- **注意**：排序是按字母顺序，不是按出现顺序！

---

### 3.5 `OneHotEncoder` — 独热编码

```python
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False)  # 新版本用 sparse_output
result = one_hot_encoder.fit_transform(data.reshape(-1, 1))  # 必须是二维！
```

| 参数 | 含义 |
|------|------|
| `sparse_output=False` | 输出普通 numpy 数组（否则输出稀疏矩阵，节省内存） |

- **输入**：**必须是二维数组**，所以要用 `.reshape(-1, 1)` 把一维列表转成列向量
- **输出**：二维 0/1 矩阵，shape 为 `(样本数, 类别数)`

##### 为什么需要 reshape？

```python
animals = ['cat', 'dog', 'fish', 'dog', 'cat']  # shape: (5,)  ← 一维
# OneHotEncoder 要求二维输入，所以要转成：
animals_2d = np.array(animals).reshape(-1, 1)    # shape: (5, 1) ← 二维列向量
```

`reshape(-1, 1)` 的含义：
- `-1`：自动计算这一维的大小（这里是 5）
- `1`：1 列

---

## 4. 运行结果示例

```
原始数据:        标准化后:        范围缩放后:      归一化后:
[[35.14  45.40]  [-0.76   0.18]  [0.27   0.44]  [0.95   0.32]
 [68.10  15.59]  [ 0.29  -1.04]  [0.65   0.00]  [0.98   0.20]
 [12.00  83.46]  [-1.50   1.72]  [0.00   1.00]  [0.55   0.83]
 [80.84  43.32]  [ 0.70   0.09]  [0.80   0.41]  [0.70   0.71]
 [98.44  17.63]  [ 1.26  -0.95]  [1.00   0.03]  [0.66   0.75]]

标签编码: [0 1 2 1 0]

独热编码:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
```

---

## 5. 常见报错与解决

| 报错信息 | 原因 | 解决 |
|---------|------|------|
| `Expected 2D array, got 1D array` | 输入了一维数组 | 用 `.reshape(-1, 1)` 转成二维 |
| `'sparse' is deprecated` | 旧参数名 | 改成 `sparse_output=False` |
| `LabelEncoder` 不接受 `sparse_output` | 误用了 OneHotEncoder 的参数 | 直接用 `LabelEncoder()`，不传参数 |

---

## 6. 一句话记忆

> **StandardScaler 对齐中心，MinMaxScaler 压缩范围，normalize 统一方向，LabelEncoder 给编号，OneHotEncoder 拆成列。**
