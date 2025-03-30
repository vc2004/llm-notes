import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.pos_encoder = PositionalEncoding(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    # 删除多余的类嵌套定义
    def forward(self, src):
        """
        参数:
            src: [batch_size, seq_len, d_model]
        返回:
            [batch_size, seq_len, d_model]
        流程:
        1. 位置编码
        2. 自注意力 + 残差连接 + 层归一化
        3. 前馈网络 + 残差连接 + 层归一化
        """
        # 输入维度应为 (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)
        
        # 保存原始维度用于残差连接
        src_orig = src
        
        # 调整维度为 (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        
        # 自注意力机制
        attn_output, attn_weights = self.self_attn(src, src, src)  # 获取注意力权重
        self.last_attn = attn_weights  # 保存供可视化
        
        # 恢复维度 (batch_size, seq_len, d_model)
        attn_output = attn_output.permute(1, 0, 2)
        
        # 残差连接
        src = src_orig + self.dropout(attn_output)
        src = self.norm(src)
        
        # 前馈网络
        ff_output = self.linear2(torch.relu(self.linear1(src)))
        src = src + self.dropout(ff_output)
        return self.norm(src)

# 删除多余的类定义结尾
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # 修改第二维度为1以支持广播
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 调整维度索引
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 维度应为 (batch_size, seq_len, d_model)
        pe = self.pe[:x.size(1)]  # 改为按seq_len维度切片
        x = x + pe.permute(1, 0, 2)  # 调整维度为 (batch_size, seq_len, d_model)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=30000, d_model=512, n_layers=6):  # 新增层数参数
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model) for _ in range(n_layers)  # 多层堆叠
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        # 逐层处理
        for encoder in self.encoder_layers:
            encoded = encoder(embedded)
        logits = self.fc(encoded)
        return torch.softmax(logits, dim=-1)

# 测试用例
if __name__ == "__main__":
    model = Transformer()
    src = torch.randint(0, 30000, (10, 32))  # (seq_len, batch_size)
    output = model(src)
    print(f"输入形状: {src.shape} => 输出形状: {output.shape}")
    print(f"样例输出概率: {output[0,0,:5].detach().numpy()}")  # 添加detach()和numpy转换
    # 可视化第一个头的注意力
    import matplotlib.pyplot as plt
    plt.matshow(model.encoder_layers[0].last_attn[0].detach().numpy())
    plt.title("Attention Weights")
    plt.show()