import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tran_dynamic_zeq import Transformer

# === 模型1和模型2路径 ===
MODEL1_PATH = "model1_single_pole.pth"
MODEL2_PATH = "model2_bias_predictor_no_mask.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 超参数 ===
embed_dim = 32
dense_dim = 64
num_heads = 4
dropout_rate = 0.1
num_blocks = 2
conv_out_channels = 16
output_sequence_length = 1
input_dim = 6  # [t, dummy, Re(p), Im(p), k, A]
batch_size = 32
lr = 1e-3
epochs = 100

# === 加载模型1和模型2 ===
model1 = Transformer(embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,
                     output_sequence_length, conv_out_channels, input_dim).to(DEVICE)
model1.load_state_dict(torch.load(MODEL1_PATH))
model1.eval()

model2 = Transformer(embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,
                     output_sequence_length, conv_out_channels, input_dim).to(DEVICE)
model2.load_state_dict(torch.load(MODEL2_PATH))
model2.eval()

# === 模型3定义：用于拟合 E_i 系列 ===
class EiPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, conv_out_channels):
        super().__init__()
        self.model = Transformer(embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,
                                 output_sequence_length=1, conv_out_channels=conv_out_channels,
                                 input_dim=input_dim)

    def forward(self, x):
        return self.model(x)  # [B, 1]

model3 = EiPredictor(input_dim=4, embed_dim=32, dense_dim=64, num_heads=4,
                     dropout_rate=0.1, num_blocks=2, conv_out_channels=16).to(DEVICE)
optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# === 加载训练数据 (每行代表一个分式) ===
df = pd.read_csv("multi_term_train.csv")

# === 数据预处理函数 ===
def prepare_samples(df):
    samples = []
    grouped = df.groupby("group_id")  # 每个 group_id 是一个完整的传递函数
    for _, group in grouped:
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 3:
            continue

        # 真实响应
        t = torch.tensor(group.loc[0, 't'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        T_true = torch.tensor(group.loc[0, 'y_true'], dtype=torch.float32).to(DEVICE)

        # R1: 只用第一个分式
        F1 = group.loc[0, ['t', 'Re_p', 'Im_p', 'k', 'A']].astype(np.float32).values
        F1_input = torch.tensor([[F1[0], 0, F1[1], F1[2], F1[3], F1[4]]], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        R1 = model1(F1_input).squeeze()

        y_pred = R1
        for i in range(1, n):
            Fi = group.loc[i, ['t', 'Re_p', 'Im_p', 'k', 'A']].astype(np.float32).values
            Fi_input = torch.tensor([[Fi[0], 0, Fi[1], Fi[2], Fi[3], Fi[4]]], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            bias_i = model2(Fi_input).squeeze()

            if i == 1:
                y_pred = R1 + bias_i
            else:
                Ei_target = (T_true - y_pred) / bias_i
                ei_input = torch.tensor([[i + 1, Fi[0], bias_i.item(), y_pred.item()]],
                                        dtype=torch.float32).unsqueeze(0).to(DEVICE)
                samples.append((ei_input, Ei_target.unsqueeze(0)))
                y_pred = y_pred + Ei_target * bias_i

    return samples

samples = prepare_samples(df)
inputs = torch.cat([s[0] for s in samples], dim=0)
targets = torch.cat([s[1] for s in samples], dim=0)
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === 训练模型3 ===
for epoch in range(epochs):
    model3.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model3(xb).squeeze(-1)
        loss = loss_fn(pred, yb.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1:03}: Loss = {total_loss / len(loader.dataset):.6f}")

# === 保存模型3 ===
torch.save(model3.state_dict(), "modelsave/model3_Ei_predictor.pth")

