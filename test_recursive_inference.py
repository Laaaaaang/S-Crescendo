# test_recursive_inference.py
import os, glob, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────── Transformer 基类 ───────────────
class StaticCondTransformer(nn.Module):
    def __init__(self, seq_dim, static_dim,
                 d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed  = nn.Linear(seq_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4,
                                         dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc, num_layers)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.head       = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x_seq, x_static):
        seq_feat  = self.encoder(self.seq_embed(x_seq))[:, -1, :]
        sta_feat  = self.static_mlp(x_static)
        return self.head(torch.cat([seq_feat, sta_feat], dim=-1)).squeeze(-1)

# ──────────── 加载一阶模型 (含 p_min / p_max) ────────────
def load_static_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(seq_dim=1, static_dim=2,
                                  d_model=cfg.get('d_model',64),
                                  nhead =cfg.get('heads',4),
                                  num_layers=cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    p_min = float(ckpt['p_min']);  p_max = float(ckpt['p_max'])
    lookback = cfg.get('lookback', 50)
    return model, p_min, p_max, lookback

# ──────────── 加载 error_n 模型 ────────────
def load_error_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(seq_dim=1, static_dim=5,
                                  d_model=cfg.get('d_model',64),
                                  nhead =cfg.get('heads',4),
                                  num_layers=cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt.get('state_dict', ckpt)); model.eval()
    return model

# ──────────── 单文件推理 ────────────
def infer_file(fp, static_model, p_min, p_max, lookback, err_model):
    df = pd.read_csv(fp)

    # 1) 时间 z‑score
    t = df['log_time'].values.astype(np.float32)
    y = df['voltage' ].values.astype(np.float32)
    t_norm = (t - t.mean()) / (t.std() + 1e-6)

    # 2) 提取并排序 (A, p)
    rc_cols = sorted([c for c in df.columns if c.startswith('RC_')],
                     key=lambda x: int(x.split('_')[1]))
    rc   = df[rc_cols].values.astype(np.float32)
    order = rc.shape[1] // 2
    A_raw = rc[:, 0::2];  p_raw = rc[:, 1::2]
    idx   = np.argsort(-p_raw[0])
    A_raw, p_raw = A_raw[:, idx], p_raw[:, idx]

    # 3) p log10+MinMax
    p_all_log = np.log10(np.abs(p_raw) + 1e-8)
    denom = (p_max - p_min) + 1e-6
    p_scaled_full = (p_all_log - p_min) / denom

    # 4) 滑窗
    N, M = len(df), len(df) - lookback
    X_seq = np.stack([t_norm[i:i+lookback] for i in range(M)])[:, :, None]

    # 5) 静态批次 [RC_0_raw, RC_1_scaled]
    rc0_raw = df['RC_0'].values.astype(np.float32)
    rc1_scaled = (np.log10(np.abs(df['RC_1'].values)+1e-8) - p_min) / denom
    X_sta = np.column_stack([rc0_raw, rc1_scaled]).astype(np.float32)
    static_batch = torch.from_numpy(X_sta[:M]).to(DEVICE)

    # 6) 一阶模型响应
    A_window = A_raw[lookback:, :]
    sum_A = A_window.sum(axis=1, keepdims=True)
    with torch.no_grad():
        base = static_model(torch.from_numpy(X_seq).to(DEVICE),
                            static_batch).cpu().numpy()
    S = (base[:, None] * (A_window / sum_A)).T  # (order,M)

    # 7) error_n
    error_sum = np.zeros(M, dtype=np.float32)
    with torch.no_grad():
        seq_batch = torch.from_numpy(X_seq).to(DEVICE)
        for n in range(2, order+1):
            raw_feat = np.column_stack([
                np.full(M, n, dtype=np.float32),
                A_raw[lookback:, n-2], p_scaled_full[lookback:, n-2],
                A_raw[lookback:, n-1], p_scaled_full[lookback:, n-1]
            ]).astype(np.float32)
            err = err_model(
                seq_batch, torch.from_numpy(raw_feat).to(DEVICE)
            ).cpu().numpy()

            # if n > 1:
            #     decay_steps = n - 1
            #     scale = (1.0 / 5.0) ** decay_steps
            #     err = scale * err

            error_sum += err

    pred  = S.sum(axis=0) + error_sum
    # pred  = S.sum(axis=0)
    times = t[lookback:]; truths = y[lookback:]
    return times, truths, pred, r2_score(truths, pred)

# ──────────── CLI ────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',      required=True)
    ap.add_argument('--static_model',  required=True)
    ap.add_argument('--err_model',     required=True)
    ap.add_argument('--out_dir',       default='results')
    args = ap.parse_args()

    static_model, p_min, p_max, lookback = load_static_model(args.static_model)
    err_model = load_error_model(args.err_model)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for fp in sorted(glob.glob(os.path.join(args.data_dir, '*.csv'))):
        start_time = time.time()

        times, truths, pred, r2 = infer_file(
            fp, static_model, p_min, p_max, lookback, err_model
        )
        fn = os.path.splitext(os.path.basename(fp))[0]
        end_time = time.time()
        print(end_time - start_time)
        # 保存 pred/real CSV
        pd.DataFrame({
            'log_time': times,
            'pred':     pred,
            'real':     truths
        }).to_csv(os.path.join(args.out_dir, f'{fn}_pred.csv'), index=False)

        # 绘图
        plt.figure(figsize=(8,4))
        plt.plot(times, truths, label='Real')
        plt.plot(times, pred,   label='Predicted')
        plt.title(f'{fn}  R²={r2:.4f}')
        plt.xlabel('log_time'); plt.ylabel('voltage')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'{fn}.png')); plt.close()

        summary.append({'file': fn, 'r2': r2})

    pd.DataFrame(summary).to_csv(
        os.path.join(args.out_dir, 'summary_r2.csv'), index=False)
    print('推理完成 — 所有结果已保存至', args.out_dir)
