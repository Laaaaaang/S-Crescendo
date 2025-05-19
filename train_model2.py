import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tran_dynamic_zeq import Transformer

# === 超参数设定 ===
embed_dim = 32
dense_dim = 64
num_heads = 4
dropout_rate = 0.1
num_blocks = 2
conv_out_channels = 16
output_sequence_length = 1
epochs = 100
batch_size = 64
lr = 1e-3
input_dim = 6  # [t, dummy, Re(p), Im(p), k, A]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 读取数据 ===
df = pd.read_csv("bias_train.csv")

# 构造输入特征: [t, dummy=0, Re(p), Im(p), k, A]
features = df[['t']].values.astype(np.float32)
dummy = np.zeros_like(features)
params = df[['Re_p', 'Im_p', 'k', 'A']].values.astype(np.float32)
features = np.concatenate([features, dummy, params], axis=1)  # [B, 6]
features = torch.tensor(features).unsqueeze(1).to(device)  # [B, 1, 6]

# 构造监督标签: bias = y_true - R1
bias_target = torch.tensor(
    df['y_true'].values - df['R1'].values,
    dtype=torch.float32
).unsqueeze(1).to(device)  # [B, 1]

# === 数据加载器 ===
dataset = TensorDataset(features, bias_target)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === 初始化模型 ===
model = Transformer(embed_dim, dense_dim, num_heads, dropout_rate, num_blocks,
                    output_sequence_length, conv_out_channels, input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# === 训练循环 ===
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        pred = model(xb).squeeze(-1)  # [B]
        loss = loss_fn(pred, yb.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1:03}: Loss = {total_loss / len(loader.dataset):.6f}")

# === 保存模型 ===
torch.save(model.state_dict(), "modelsave/model2_bias_predictor_no_mask.pth")
