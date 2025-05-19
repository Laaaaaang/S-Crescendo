# combine_and_train_error_model.py
import os, glob, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ───────────────── Static‑Cond Transformer ─────────────────
class StaticCondTransformer(nn.Module):
    def __init__(self, seq_dim=1, static_dim=5,
                 d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed = nn.Linear(seq_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                         dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x_seq, x_static):
        seq_feat  = self.encoder(self.seq_embed(x_seq))[:, -1, :]  # (B,D)
        stat_feat = self.static_mlp(x_static)                      # (B,D)
        return self.head(torch.cat([seq_feat, stat_feat], dim=-1)).squeeze(-1)

# ───────────────── 构造滑窗 ─────────────────
def build_windows(df: pd.DataFrame, lookback: int):
    """
    返回：
        X_seq, X_sta, Y, times, p_min, p_max
    """
    # 时间归一化
    t    = df['t'].values.astype(np.float32)
    err  = df['error'].values.astype(np.float32)
    t_norm = (t - t.mean()) / (t.std() + 1e-6)

    # 静态特征
    cols = ['n','A_prev','p_prev','A_new','p_new']
    raw  = df[cols].values.astype(np.float32)
    n, A_prev, p_prev, A_new, p_new = raw.T

    # log10 + MinMax 缩放 p
    p_all = np.log10(np.abs(np.concatenate([p_prev, p_new])) + 1e-8)
    p_min, p_max = p_all.min(), p_all.max()
    p_prev_s = (np.log10(np.abs(p_prev)+1e-8)-p_min)/(p_max-p_min+1e-6)
    p_new_s  = (np.log10(np.abs(p_new) +1e-8)-p_min)/(p_max-p_min+1e-6)

    static_norm = np.stack([n, A_prev, p_prev_s, A_new, p_new_s], axis=1)
    print("static norm",static_norm)

    # 滑窗
    N = len(df)
    X_seq, X_sta, Y, times = [], [], [], []
    for i in range(N - lookback):
        X_seq.append(t_norm[i:i+lookback, None])
        X_sta.append(static_norm[i+lookback])
        Y.append(err[i+lookback])
        times.append(t[i+lookback])

    return (np.stack(X_seq).astype(np.float32),
            np.stack(X_sta).astype(np.float32),
            np.array(Y, dtype=np.float32),
            np.array(times, dtype=np.float32),
            float(p_min), float(p_max))

# ───────────────── 数据块划分 ─────────────────
def chunked_split(N, chunk_size=1000, train_frac=0.9):
    idx = np.arange(N); tr, te = [], []
    for s in range(0, N, chunk_size):
        block = idx[s: min(s+chunk_size, N)]
        np.random.shuffle(block)
        cut = int(len(block)*train_frac)
        tr.extend(block[:cut]); te.extend(block[cut:])
    return np.array(tr), np.array(te)

# ─────────────────── main ───────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--pattern", default="error*_dataset.csv")
    p.add_argument("--lookback", type=int, default=50)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_model", default="models/model_error_n.pth")
    p.add_argument("--out_preds", default="models/test_predictions.csv")
    args = p.parse_args()

    # 1) 合并所有 CSV
    paths = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError("No CSV matched")
    merged = pd.concat([pd.read_csv(fp) for fp in paths], ignore_index=True)
    print(f"🔗 merged rows: {len(merged)}")

    # 2) 构造窗口
    X_seq, X_sta, Y, times, p_min, p_max = build_windows(merged, args.lookback)
    print(f"windows: {len(Y)}  p_min={p_min:.3e}  p_max={p_max:.3e}")

    # 3) 切分
    tr_idx, te_idx = chunked_split(len(Y))
    te_idx = te_idx[np.argsort(times[te_idx])]

    dl_tr = DataLoader(TensorDataset(
        torch.from_numpy(X_seq[tr_idx]),
        torch.from_numpy(X_sta[tr_idx]),
        torch.from_numpy(Y[tr_idx])), batch_size=args.batch_size, shuffle=True)
    dl_te = DataLoader(TensorDataset(
        torch.from_numpy(X_seq[te_idx]),
        torch.from_numpy(X_sta[te_idx]),
        torch.from_numpy(Y[te_idx])), batch_size=args.batch_size)

    # 4) 训练
    model = StaticCondTransformer().to(DEVICE)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=args.lr), nn.MSELoss()
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0.0
        for xb, xs, yb in dl_tr:
            xb, xs, yb = xb.to(DEVICE), xs.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb, xs), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        print(f"Epoch {ep}/{args.epochs}  loss={tot/len(tr_idx):.4e}")

    # 5) 测试预测
    model.eval(); pr, gt = [], []
    with torch.no_grad():
        for xb, xs, yb in dl_te:
            xb, xs = xb.to(DEVICE), xs.to(DEVICE)
            pr.append(model(xb, xs).cpu().numpy()); gt.append(yb.numpy())
    pred, true = np.concatenate(pr), np.concatenate(gt)
    pd.DataFrame({"time": times[te_idx], "true": true, "pred": pred}
                ).to_csv(args.out_preds, index=False)
    print("✓ test preds saved →", args.out_preds)

    # 6) 保存模型 + p_min/p_max
    cfg = vars(args)
    os.makedirs(os.path.dirname(args.out_model) or '.', exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "p_min": p_min,
        "p_max": p_max,
        "cfg": cfg
    }, args.out_model)
    print("✓ model saved →", args.out_model)

# python combine_and_train_error_model.py  --csv_dir data/error   --pattern 'error*_dataset.csv'   --lookback 50   --epochs 50   --batch_size 1000   --lr 1e-3   --out_model models/model_error_n.pth --out_preds models/test_predictions.csv