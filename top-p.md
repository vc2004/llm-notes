`top-p`参数（也称为核采样或Nucleus Sampling）用于控制文本生成过程中候选词的多样性，其作用如下：

### 核心功能
- **动态候选词选择**：从概率最高的词开始累加，直到累积概率超过阈值`p`，仅从这些词中随机选择下一个词。例如，若`p=0.9`，模型会选择累积概率达90%的最小词集合。

### 关键特点
1. **平衡多样性与相关性**：
   - **低`p`值**（如0.5）：候选词集合小，生成内容更保守、连贯，但可能重复。
   - **高`p`值**（如0.95）：候选词更多样，生成结果更创意，但可能偏离上下文。

2. **自适应调整**：相比固定`top-k`采样，`top-p`根据当前词的概率分布动态调整候选词数量。概率分布尖锐时选词少，平坦时选词多。

3. **与温度参数协同**：
   - **温度（Temperature）**：调整概率分布的平滑度（高温更均匀，低温更尖锐）。
   - **组合使用**：高温+高`top-p`可极大增加多样性；低温+低`top-p`则生成更确定的内容。

### 典型应用场景
- **创意文本**（如故事、诗歌）：`p=0.9~1.0`，鼓励多样性。
- **事实性内容**（如问答、摘要）：`p=0.5~0.8`，保持准确性。
- **代码生成**：中等`p`值（如0.7~0.9），平衡语法正确性与灵活性。

### 示例
假设生成下一个词的候选概率为：`猫(0.4)`, `狗(0.3)`, `车(0.2)`, `跑(0.1)`：
- 若`p=0.7`，累积概率为`猫(0.4)→狗(0.7)`，仅从“猫”和“狗”中选择。
- 若`p=0.95`，需加入“车”达到累积0.9，仍不足，继续加“跑”至1.0，此时从所有词中选择。

### 注意事项
- **避免极端值**：`p=1.0`可能导致不相关词被选，`p≈0`可能退化为贪婪搜索（选最高概率词）。
- **调试建议**：从`p=0.9`开始，根据输出结果微调，观察多样性与连贯性的平衡。

总结：`top-p`通过动态控制候选词范围，帮助生成模型在“创造性”与“准确性”之间取得平衡，是调整文本风格的重要参数之一。