在Transformer的Decoder中，**Autoregressive（自回归）** 是指生成序列时，模型只能基于**已生成的部分结果**逐步预测下一个输出，且每个步骤的输出依赖于之前生成的所有内容。这种机制是序列生成任务（如机器翻译、文本生成）的核心特性。

---

### **核心概念解释**
1. **自回归的字面含义**：
   - **Auto（自）**：生成过程是自我驱动的，每一步的输入是上一步的输出。
   - **Regression（回归）**：统计学中“回归”指预测连续值，这里指预测序列中的下一个元素。

2. **在Transformer Decoder中的具体实现**：
   - **训练阶段**：Decoder通过**Teacher Forcing**接收真实的标签序列作为输入（例如，翻译任务的正确目标句子），但会使用**Masked Self-Attention**隐藏未来位置的信息，强制模型仅依赖已生成的内容。
   - **推理阶段**：Decoder逐步生成序列，每一步生成一个词，并将其作为下一步的输入（例如：生成第一个词后，用这个词预测第二个词，依此类推）。

---

### **自回归的直观例子**
以生成句子 `"I love cats"` 为例：
1. 输入起始符 `<sos>`，模型输出第一个词 `"I"`。
2. 输入 `<sos> I`，模型输出第二个词 `"love"`。
3. 输入 `<sos> I love`，模型输出第三个词 `"cats"`。
4. 输入 `<sos> I love cats`，模型输出结束符 `<eos>`。

每一步的预测严格依赖于之前的输出，形成一个链式依赖。

---

### **技术细节**
1. **Masked Self-Attention**：
   - Decoder的自注意力层会通过**掩码（Mask）**阻止当前位置关注未来的位置。例如，生成第3个词时，模型只能看到第1、2个词的信息。
   - 这是自回归的核心实现，确保模型在训练和推理时行为一致。

2. **位置编码（Positional Encoding）**：
   - 自回归生成依赖顺序，因此Decoder需要位置编码来明确每个词的位置信息。

3. **与Encoder的差异**：
   - Encoder可以看到完整的输入序列（无掩码），而Decoder只能看到已生成的部分。

---

### **自回归的优缺点**
- **优点**：
  - 生成质量高：逐步生成可以利用历史信息，保持一致性。
  - 适合开放域生成（如对话、故事生成）。

- **缺点**：
  - **无法并行生成**：必须逐词生成，速度较慢（例如GPT-3生成长文本耗时较长）。
  - 错误累积：早期生成的错误会影响后续结果。

---

### **非自回归模型（对比）**
为了解决自回归的缺陷，研究者提出了**非自回归模型（Non-Autoregressive）**，允许同时生成所有位置的词（例如用于语音合成）。但这类模型通常需要额外设计（如迭代修正）来保证生成质量。

---

### **总结**
自回归是Transformer Decoder的核心机制，通过逐步生成序列并严格依赖历史信息，确保了生成结果的连贯性和合理性。虽然牺牲了生成速度，但仍然是目前大多数生成任务（如GPT、机器翻译）的首选方案。