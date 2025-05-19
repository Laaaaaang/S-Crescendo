# train_error_model.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class StaticCondTransformer(nn.Module):
    def __init__(self, seq_dim=1, static_dim=5,
                 d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed = nn.Linear(seq_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_seq, x_static):
        # x_seq: (B,L,1), x_static: (B,5)
        seq = self.seq_embed(x_seq)
        seq = self.encoder(seq)
        seq_feat = seq[:, -1, :]
        sta_feat = self.static_mlp(x_static)
        return self.head(torch.cat([seq_feat, sta_feat], dim=-1)).squeeze(-1)


def build_dataset(df: pd.DataFrame, lookback: int):
    # 1) 时间 t 归一化（文件级）
    t = df['t'].values.astype(np.float32)
    mu_t, sd_t = t.mean(), t.std() + 1e-6
    t_norm = (t - mu_t) / sd_t

    # 2) 静态特征归一化：仅对 p 做 log10+MinMax，其他保留原始值
    cols = ['n', 'A_prev', 'p_prev', 'A_new', 'p_new']
    sta_raw = df[cols].values.astype(np.float32)  # (N,5)
    
    # 提取各列
    n_col      = sta_raw[:, 0]
    A_prev_col = sta_raw[:, 1]
    p_prev_col = sta_raw[:, 2]
    A_new_col  = sta_raw[:, 3]
    p_new_col  = sta_raw[:, 4]

    # 仅对 p 列进行 log10 + MinMax 归一化
    p_all_log = np.log10(np.abs(np.concatenate([p_prev_col, p_new_col])) + 1e-8)
    p_min = p_all_log.min()
    p_max = p_all_log.max()
    
    p_prev_scaled = (np.log10(np.abs(p_prev_col) + 1e-8) - p_min) / (p_max - p_min + 1e-6)
    p_new_scaled  = (np.log10(np.abs(p_new_col)  + 1e-8) - p_min) / (p_max - p_min + 1e-6)

    # 拼接成最终静态特征矩阵
    sta_norm = np.stack([n_col, A_prev_col, p_prev_scaled, A_new_col, p_new_scaled], axis=1)

    # 3) 目标 error
    err = df['error'].values.astype(np.float32)

    # 4) 滑窗构造
    N = len(df)
    X_seq, X_sta, Y, times = [], [], [], []
    for i in range(N - lookback):
        X_seq.append(t_norm[i:i+lookback, None])    # (lookback,1)
        X_sta.append(sta_norm[i+lookback])          # (5,)
        Y.append(err[i+lookback])
        times.append(t[i+lookback])
    return (np.stack(X_seq).astype(np.float32),
            np.stack(X_sta).astype(np.float32),
            np.array(Y, dtype=np.float32),
            np.array(times, dtype=np.float32))



def chunked_split(N, chunk_size=1000, train_frac=0.9):
    idx = np.arange(N)
    tr, te = [], []
    for start in range(0, N, chunk_size):
        block = idx[start: min(start+chunk_size, N)]
        np.random.shuffle(block)
        cut = int(len(block)*train_frac)
        tr.extend(block[:cut])
        te.extend(block[cut:])
    return np.array(tr), np.array(te)


def train(csv_path, epochs=40, lookback=50, batch=256, lr=1e-3, model_out='models/model_error_n.pth'):
    # 读取并构建数据
    df = pd.read_csv(csv_path)
    X_seq, X_sta, Y, times = build_dataset(df, lookback)
    M = len(Y)
    print(f"[train_error_model] windows: {M}")

    # 切分训练/测试
    tr_idx, te_idx = chunked_split(M, chunk_size=1000, train_frac=0.9)
    # 测试集按时间排序
    te_idx = te_idx[np.argsort(times[te_idx])]

    ds_tr = TensorDataset(torch.from_numpy(X_seq[tr_idx]),
                          torch.from_numpy(X_sta[tr_idx]),
                          torch.from_numpy(Y[tr_idx]))
    ds_te = TensorDataset(torch.from_numpy(X_seq[te_idx]),
                          torch.from_numpy(X_sta[te_idx]),
                          torch.from_numpy(Y[te_idx]))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch)

    # 模型/优化/损失
    model = StaticCondTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 训练循环
    for ep in range(1, epochs+1):
        model.train()
        tot = 0.0
        for xb, xs, yb in dl_tr:
            xb, xs, yb = xb.to(DEVICE), xs.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, xs)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"Epoch {ep}/{epochs}  train_loss={tot/len(ds_tr):.4e}")

    # 保存模型
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"✓ model saved: {model_out}")

    # 测试预测并保存
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, xs, yb in dl_te:
            xb, xs = xb.to(DEVICE), xs.to(DEVICE)
            p = model(xb, xs).cpu().numpy()
            all_p.append(p); all_y.append(yb.numpy())
    pred = np.concatenate(all_p)
    true = np.concatenate(all_y)
    times_te = times[te_idx]

    out_preds = model_out.replace('.pth', '_test_preds.csv')
    pd.DataFrame({
        'time': times_te,
        'true_error': true,
        'pred_error': pred
    }).to_csv(out_preds, index=False)
    print(f"✓ test predictions saved: {out_preds}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True, help='error_k_dataset.csv')
    ap.add_argument('--lookback', type=int,   default=50)
    ap.add_argument('--epochs',   type=int,   default=40)
    ap.add_argument('--batch',    type=int,   default=64)
    ap.add_argument('--lr',       type=float, default=3e-4)
    ap.add_argument('--out',      default='models/model_error_n.pth')
    args = ap.parse_args()

    train(args.csv, args.epochs, args.lookback, args.batch, args.lr, args.out)
