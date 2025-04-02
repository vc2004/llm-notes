在Transformer架构中，**Token Embedding**（词嵌入）是将离散的输入符号（如单词、子词或字符）转换为连续向量表示的核心步骤。以下是其详细解释：

---

### **1. Tokenization（分词）**
- **输入处理**：原始文本首先被分割成更小的单元（tokens）。例如：
  - **单词级**：`"Hello world"` → `["Hello", "world"]`
  - **子词级**（如WordPiece或BPE）：`"unhappy"` → `["un", "happy"]`
  - **字符级**：`"A"` → `["A"]`
- **映射到索引**：每个token被分配一个唯一的整数索引（基于预定义的词汇表）。例如：
  - `"Hello"` → 索引`5`
  - `"world"` → 索引`12`

---

### **2. 嵌入矩阵（Embedding Matrix）**
- **矩阵定义**：模型维护一个可训练的矩阵（参数），形状为 `[vocab_size, d_model]`：
  - `vocab_size`：词汇表大小（所有可能的token数量）。
  - `d_model`：嵌入向量的维度（如512）。
- **查找操作**：通过索引从矩阵中提取对应的向量。例如：
  - 索引`5` → 提取第5行的向量（形状`[1, 512]`）。
  - 索引`12` → 提取第12行的向量（形状`[1, 512]`）。

---

### **3. 数学表示**
- 输入序列的每个token索引 `i`，映射为向量：
  ```
  embedding_vector = embedding_matrix[i]
  ```
- 输入序列`[token_1, token_2, ..., token_n]` → 输出形状为 `[n, d_model]` 的矩阵。

---

### **4. 核心作用**
- **语义表示**：通过训练，模型学习将语义相似的token映射到邻近的向量空间。例如：
  - `"cat"`和`"dog"`的向量距离较近，而`"cat"`和`"computer"`的向量距离较远。
- **降维**：将高维的稀疏one-hot向量（维度=词汇表大小）压缩为低维的稠密向量（如512维），便于模型高效处理。

---

### **5. 实现示例（PyTorch）**
```python
import torch
import torch.nn as nn

# 定义嵌入层
vocab_size = 10000  # 假设词汇表含10,000个token
d_model = 512
embedding = nn.Embedding(vocab_size, d_model)

# 输入：一个batch的token索引（假设batch_size=2，序列长度=3）
input_indices = torch.LongTensor([[5, 12, 8], [1, 3, 0]])  # 形状 [2, 3]

# 获取嵌入向量
output = embedding(input_indices)  # 形状 [2, 3, 512]
```

---

### **6. 关键细节**
- **可训练参数**：嵌入矩阵是模型的一部分，通过反向传播更新。
- **初始化**：通常使用随机初始化（如正态分布），某些场景会预加载预训练词向量（如GloVe）。
- **与后续步骤的衔接**：Token Embedding的输出会与位置编码相加，再输入到Transformer的编码器中。

---

### **常见问题**
#### **Q1：为什么需要Token Embedding？**
- **数值化**：将离散符号转换为连续数值，使模型能处理文本。
- **语义压缩**：低维向量能捕捉语义关系（如相似性、类比关系）。

#### **Q2：如何处理未登录词（OOV）？**
- 子词分词（如WordPiece、BPE）可将未知词拆分为已知子词。
- 预留特殊token（如`[UNK]`）表示未知词。

#### **Q3：嵌入维度（d_model）如何选择？**
- 维度越大，表达能力越强，但计算成本越高（需权衡模型性能和效率）。

---

通过Token Embedding，Transformer将离散的文本符号转化为富含语义信息的向量，为后续的自注意力机制和深度计算奠定了基础。