# train_model1.py
"""
训练一阶模型（静态分支：RC_0 原值、RC_1 log10+MinMax）
---------------------------------------------------------
1. 全局拼接 RC_cated_*.csv
2. RC_0 不做归一化；RC_1 用 log10+MinMax
3. 保存 p_min / p_max 供推理阶段使用
"""

import os, glob, argparse, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------  模型定义 ------------------
class StaticCondTransformer(nn.Module):
    def __init__(self,
                 seq_dim=1, static_dim=2,
                 d_model=64, nhead=4,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed = nn.Linear(seq_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                         dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(),
            nn.Linear(d_model, 1))

    def forward(self, x_seq, x_static):
        seq_feat = self.encoder(self.seq_embed(x_seq))[:, -1, :]
        sta_feat = self.static_mlp(x_static)
        return self.head(torch.cat([seq_feat, sta_feat], dim=-1)).squeeze(-1)

# ------------------  数据集构建 ------------------
def build_dataset(csv_files, lookback):
    """
    返回:
        X_seq  (M,L,1)   — z‑score(log_time)
        X_sta  (M,2)     — [RC_0_raw, RC_1_scaled]
        Y      (M,)
        p_min, p_max     — log10(abs(RC_1)) 全局 min/max
    """
    rc0_all, p_all_log = [], []
    dfs = []

    # 1) 读取所有文件，收集 RC_0 / RC_1
    for fp in csv_files:
        df = pd.read_csv(fp, usecols=['log_time', 'RC_0', 'RC_1', 'voltage'])
        dfs.append(df)
        rc0_all.append(df['RC_0'].values)
        p_all_log.append(np.log10(np.abs(df['RC_1'].values) + 1e-8))

    p_all_log = np.hstack(p_all_log)
    p_min, p_max = p_all_log.min(), p_all_log.max()

    X_seq_list, X_sta_list, Y_list = [], [], []

    # 2) 对每个文件生成滑窗样本
    for df in dfs:
        rc0_raw = df['RC_0'].values.astype(np.float32)
        rc1_raw = df['RC_1'].values.astype(np.float32)
        rc1_scaled = (np.log10(np.abs(rc1_raw) + 1e-8) - p_min) / (p_max - p_min + 1e-6)
        static_feat = np.column_stack([rc0_raw, rc1_scaled])  # (N,2)

        t = df['log_time'].values.astype(np.float32)
        t_norm = (t - t.mean()) / (t.std() + 1e-6)
        seq_feat = t_norm[:, None]
        y = df['voltage'].values.astype(np.float32)

        N = len(df)
        for i in range(N - lookback):
            X_seq_list.append(seq_feat[i:i+lookback])
            X_sta_list.append(static_feat[i])
            Y_list.append(y[i + lookback])

    X_seq = np.stack(X_seq_list).astype(np.float32)
    X_sta = np.stack(X_sta_list).astype(np.float32)
    Y     = np.array(Y_list, dtype=np.float32)
    return X_seq, X_sta, Y, p_min, p_max

# ------------------  训练流程 ------------------
def train(args):
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, 'RC_cated_*.csv')))
    assert csv_files, 'No CSV found in data_dir'
    print(f'Found {len(csv_files)} csv files')

    X_seq, X_sta, Y, p_min, p_max = build_dataset(csv_files, args.lookback)
    print(f'Dataset windows: {len(Y):,}')

    # 90/10 随机拆分
    idx = np.random.permutation(len(Y))
    split = int(0.9 * len(Y))
    tr_idx, val_idx = idx[:split], idx[split:]

    ds_tr = TensorDataset(torch.tensor(X_seq[tr_idx]),
                          torch.tensor(X_sta[tr_idx]),
                          torch.tensor(Y[tr_idx]))
    ds_val= TensorDataset(torch.tensor(X_seq[val_idx]),
                          torch.tensor(X_sta[val_idx]),
                          torch.tensor(Y[val_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    dl_val= DataLoader(ds_val, batch_size=args.batch)

    model = StaticCondTransformer(d_model=args.d_model,
                                  nhead=args.heads,
                                  num_layers=args.layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for xs, ss, ys in dl_tr:
            xs, ss, ys = xs.to(DEVICE), ss.to(DEVICE), ys.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xs, ss), ys)
            loss.backward(); opt.step()
            tot += loss.item() * len(xs)
        tot /= len(ds_tr)

        # 验证
        model.eval(); val = 0.0
        with torch.no_grad():
            for xs, ss, ys in dl_val:
                xs, ss, ys = xs.to(DEVICE), ss.to(DEVICE), ys.to(DEVICE)
                val += loss_fn(model(xs, ss), ys).item() * len(xs)
        val /= len(ds_val)
        if ep == 1 or ep % 5 == 0:
            print(f'Epoch {ep}/{args.epochs}  train {tot:.4e}  val {val:.4e}')

    os.makedirs('models', exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'p_min': p_min,
        'p_max': p_max,
        'cfg': vars(args)
    }, 'models/static_cond_model.pth')
    print('✓ Model saved to models/static_cond_model.pth')
    print(f'   p_min={p_min:.4f}  p_max={p_max:.4f}')

# ------------------  CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/order1')
    p.add_argument('--lookback', type=int, default=50)
    p.add_argument('--batch',    type=int, default=64)
    p.add_argument('--epochs',   type=int, default=60)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--d_model',  type=int, default=64)
    p.add_argument('--heads',    type=int, default=4)
    p.add_argument('--layers',   type=int, default=3)
    args = p.parse_args()
    train(args)
