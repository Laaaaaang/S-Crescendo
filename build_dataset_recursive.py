# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────── 模型定义 ───────────────────
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
        seq_feat = self.encoder(self.seq_embed(x_seq))[:, -1, :]
        sta_feat = self.static_mlp(x_static)
        return self.head(torch.cat([seq_feat, sta_feat], dim=-1)).squeeze(-1)

# ─────────────────── 加载一阶模型 ───────────────────
def load_model1(path: str):
    """
    返回:
        model      : 已加载权重
        p_min/max  : 训练阶段保存的 RC_1 log10(abs) 的全局 min/max
        lookback   : 窗口长度
    """
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(seq_dim=1, static_dim=2,
                                  d_model    = cfg.get('d_model',64),
                                  nhead      = cfg.get('heads',4),
                                  num_layers = cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    p_min = float(ckpt['p_min']);  p_max = float(ckpt['p_max'])
    lookback = cfg.get('lookback', 50)
    return model, p_min, p_max, lookback

# ─────────────────── error_k teacher 加载 ───────────────────
def load_error_model(k: int, models_dir: str):
    pth = os.path.join(models_dir, f'model_error_{k}.pth')
    if not os.path.isfile(pth):
        return None
    ckpt = torch.load(pth, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(seq_dim=1, static_dim=5,
                                  d_model    = cfg.get('d_model',64),
                                  nhead      = cfg.get('heads',4),
                                  num_layers = cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt.get('state_dict', ckpt)); model.eval()
    return model

# ──────────── 一阶分式响应预测 ─────────────────
def predict_S(model1,
              t_raw: np.ndarray,
              A: float,
              static_feat: np.ndarray,
              lookback: int,
              model1_has_coef=False):
    """
    返回与 t_raw 同长度的 S_i 序列（前 lookback 补 0）
    """
    N = len(t_raw)
    t_norm = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)
    X_seq = np.stack([t_norm[i:i+lookback] for i in range(N - lookback)])[:, :, None]
    X_sta = np.repeat(static_feat[None, :], N - lookback, axis=0)

    with torch.no_grad():
        pred = model1(torch.from_numpy(X_seq).to(DEVICE),
                      torch.from_numpy(X_sta).to(DEVICE)).cpu().numpy()

    pred = pred if model1_has_coef else (A * pred)
    return np.concatenate([np.zeros(lookback, dtype=np.float32), pred])

# ──────────── error_n 递归预测 ─────────────────
def predict_error_n(model_error, lookback,
                    t_raw: np.ndarray,
                    static_feat: np.ndarray):
    N = len(t_raw)
    t_norm = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)
    x_seq = torch.tensor(
        np.stack([t_norm[i:i+lookback] for i in range(N - lookback)])[:, :, None],
        dtype=torch.float32, device=DEVICE)
    x_sta = torch.tensor(
        np.repeat(static_feat[None, :], N - lookback, axis=0),
        dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = model_error(x_seq, x_sta).cpu().numpy()
    return np.concatenate([np.zeros(lookback, dtype=np.float32), pred])

# ──────────── 数据集构建 ─────────────────
def build_dataset_recursive(csv_path, model1_path,
                            err_dir, out_csv,
                            model1_has_coef=False):
    model1, p_min, p_max, lookback = load_model1(model1_path)

    df     = pd.read_csv(csv_path)
    t_raw  = df['log_time'].values.astype(np.float32)
    y_true = df['voltage' ].values.astype(np.float32)

    # --- 静态特征预处理：RC_0 原值，RC_1 log10+MinMax ---
    rc0_raw = df['RC_0'].values.astype(np.float32)
    rc1_raw = df['RC_1'].values.astype(np.float32)
    rc1_scaled = (np.log10(np.abs(rc1_raw) + 1e-8) - p_min) / (p_max - p_min + 1e-6)
    static_two_scaled = np.column_stack([rc0_raw, rc1_scaled]).astype(np.float32)

    # --- 提取 (A, p) 并按数值降序排序 ---
    cols   = [c for c in df.columns if c.startswith('RC_')]
    order  = len(cols) // 2
    A_list = [float(df[f'RC_{2*i}'][0])   for i in range(order)]
    p_list = [float(df[f'RC_{2*i+1}'][0]) for i in range(order)]
    idx    = sorted(range(order), key=lambda i: -p_list[i])
    A_sorted = [A_list[i] for i in idx]
    p_sorted = [p_list[i] for i in idx]

    # --- p_scaled 供 error_n ---
    p_log_all = np.log10(np.abs(np.array(p_sorted)) + 1e-8)
    p_scaled = (p_log_all - p_min) / (p_max - p_min + 1e-6)

    # --- 一阶分式响应 ---
    S = []
    for i in range(order):
        S_i = predict_S(model1, t_raw,
                        A_sorted[i],
                        static_two_scaled[0],   # 首行静态
                        lookback,
                        model1_has_coef)
        S.append(S_i)
    S = np.stack(S, axis=0)

    # --- error_n 残差 ---
    error_sum = np.zeros_like(t_raw, dtype=np.float32)
    for k in range(2, order):
        teacher = load_error_model(k, err_dir)
        if teacher is None:
            raise RuntimeError(f"Missing model_error_{k}.pth")
        raw_feat = np.array([k,
                             A_sorted[k-2], p_scaled[k-2],
                             A_sorted[k-1], p_scaled[k-1]], dtype=np.float32)
        error_sum += predict_error_n(teacher, lookback, t_raw, raw_feat)

    approx = S.sum(axis=0) + error_sum
    e_N = y_true - approx

    # 写入从 lookback 开始的有效样本
    rows = [[e_N[i], t_raw[i], order,
             A_sorted[order-2], p_sorted[order-2],
             A_sorted[order-1], p_sorted[order-1]]
            for i in range(lookback, len(t_raw))]

    pd.DataFrame(rows,
                 columns=['error','t','n',
                          'A_prev','p_prev','A_new','p_new']
                ).to_csv(out_csv, index=False)
    print(f'Wrote {len(rows)} rows → {out_csv}')

# ──────────── CLI ────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True)
    ap.add_argument('--model1',   required=True)
    ap.add_argument('--err_dir',  default='error_models')
    ap.add_argument('--out_csv',  default='modelk_dataset.csv')
    ap.add_argument('--model1_has_coef', action='store_true')
    args = ap.parse_args()

    build_dataset_recursive(args.csv, args.model1,
                            args.err_dir, args.out_csv,
                            args.model1_has_coef)
