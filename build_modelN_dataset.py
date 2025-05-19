# -*- coding: utf-8 -*-
"""
build_modelN_dataset.py  ▸  生成『模型 n』训练数据 (通用脚本)
==================================================================
输入二阶 / 三阶 / … CSV：
    t, voltage, RC_0, RC_1, RC_2, RC_3, ... , RC_{2n-2}, RC_{2n-1}
        └─ n 个分式   (coef 0, pole 0,  coef 1, pole 1, ...)

步骤 (针对 **n** 阶文件 → 训练 **模型 n**):
1. 逐个分式调用 **模型 1** (StaticCondTransformer) 得到原始预测 Pi_raw(t)
2. 按系数比例:  w_i = coef_i / Σcoef  →  S_i = w_i * Pi_raw
3. M1 = Σ S_i，error = y_true – M1
4. 只保留 **新增分式** (第 n 个) 的特征作为输入：
   写行 `[error, t, n, coef_n, pole_n]`

运行示例 (三阶 → 模型 3):
```bash
python build_modelN_dataset.py \
    --data_dir data/third_order \
    --model1 models/static_cond_model.pth \
    --order 3 \
    --out_csv model3_ds.csv
```
"""

import os, glob, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from collections import namedtuple

# ---------------- 0)  StaticCondTransformer 定义 ----------------
class StaticCondTransformer(nn.Module):
    def __init__(self, seq_dim=1, static_dim=2, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed = nn.Linear(seq_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.static_mlp = nn.Sequential(nn.Linear(static_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x_seq, x_static):
        seq_feat = self.encoder(self.seq_embed(x_seq))[:, -1, :]
        static_feat = self.static_mlp(x_static)
        return self.head(torch.cat([seq_feat, static_feat], -1)).squeeze(-1)

# ---------------- 1) 载入模型 1 & scaler ------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Bundle = namedtuple('Bundle', ['model', 'mean', 'scale'])

def load_model1(path: str) -> Bundle:
    ckpt = torch.load(path, map_location=DEVICE)
    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(d_model=cfg.get('d_model',64),
                                  nhead=cfg.get('heads',4),
                                  num_layers=cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    mean  = ckpt['scaler_mean'].astype(np.float32)
    scale = ckpt['scaler_scale'].astype(np.float32)
    return Bundle(model, mean, scale)

# ---------------- 2) 单分式推理 -------------------------------

def predict_first_order(bundle: Bundle, t_raw: np.ndarray, frac: np.ndarray):
    """t_raw (T,), frac (2,) → pred (T,)"""
    t_norm = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)
    X_seq = torch.tensor(t_norm[None, :, None], dtype=torch.float32, device=DEVICE)
    static_norm = (frac - bundle.mean) / bundle.scale
    X_sta = torch.tensor(static_norm[None, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = bundle.model(X_seq, X_sta).cpu().numpy()[0]
    return out

# ---------------- 3) 构建数据集 -------------------------------

def build_dataset(data_dir: str, model1_path: str, order: int, out_csv: str):
    bundle = load_model1(model1_path)
    rows = []

    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise RuntimeError('No csv found')

    for fp in files:
        df = pd.read_csv(fp)
        # 检查所需列
        need_cols = {'log_time', 'voltage'} | {f'RC_{i}' for i in range(2*order)}
        if not need_cols.issubset(df.columns):
            raise ValueError(f'{fp} 缺列: {need_cols - set(df.columns)}')

        t_raw  = df['log_time'].values.astype(np.float32)
        y_true = df['voltage'].values.astype(np.float32)

        # 收集所有分式 (coef, pole)
        fracs = [np.array([df[f'RC_{2*i}'][0], df[f'RC_{2*i+1}'][0]], dtype=np.float32) for i in range(order)]
        coefs = np.array([f[0] for f in fracs], dtype=np.float32)
        coef_sum = coefs.sum() if coefs.sum()!=0 else 1.0

        # 1)‑2) 预测 & 加权
        preds = [predict_first_order(bundle, t_raw, f) for f in fracs]  # list of (T,)
        weighted = [c/coef_sum * p for c,p in zip(coefs, preds)]
        M1 = np.sum(weighted, axis=0)                                  # (T,)

        err = y_true - M1                                              # (T,)

        # 新增分式 (最后一个) 的特征
        new_coef, new_pole = fracs[-1]
        dim_ext = order
        for e, tt in zip(err, t_raw):
            rows.append([e, tt, dim_ext, new_coef, new_pole])

    df_out = pd.DataFrame(rows, columns=['error', 'log_time', 'dim_ext', 'A_new', 'p_new'])
    df_out.to_csv(out_csv, index=False)
    print(f'✓ 生成 {len(df_out)} 行 → {out_csv}')

# ---------------- 4) CLI -------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser('Build dataset for Model‑n (generic)')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model1',   required=True)
    ap.add_argument('--order', type=int, required=True, help='number of fractions (n)')
    ap.add_argument('--out_csv', default='modelN_ds.csv')
    args = ap.parse_args()
    build_dataset(args.data_dir, args.model1, args.order, args.out_csv)

# python build_modelN_dataset.py --data_dir data/third_order --model1 models/static_cond_model.pth --order 3 --out_csv model3_ds.csv