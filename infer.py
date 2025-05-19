"""
Updated inference script for StaticCondTransformer (fixed dtype issue)
--------------------------------------------------
Run example:
    python infer.py --data_dir ./ --model models/static_cond_model.pth --start 81 --end 100
Changes vs previous version:
    • Ensure all numpy tensors cast to float32 **after** arithmetic, so torch tensors become Float
      → avoids "mat1 and mat2 must have the same dtype, but got Double and Float".
"""

import os, argparse, glob, numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -------------------------- model (same as training) --------------------------
class StaticCondTransformer(nn.Module):
    def __init__(self, seq_dim=1, static_dim=2, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.seq_embed = nn.Linear(seq_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.static_mlp = nn.Sequential(nn.Linear(static_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x_seq, x_static):
        seq = self.encoder(self.seq_embed(x_seq))           # (B,L,D)
        seq_feat   = seq[:, -1, :]
        static_feat= self.static_mlp(x_static)
        fused = torch.cat([seq_feat, static_feat], dim=-1)
        return self.head(fused).squeeze(-1)

# -------------------------- load model & scaler ------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt.get('cfg', {})
    model = StaticCondTransformer(d_model=cfg.get('d_model',64),
                                  nhead=cfg.get('heads',4),
                                  num_layers=cfg.get('layers',3)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    mean  = ckpt['scaler_mean'].astype(np.float32)
    scale = ckpt['scaler_scale'].astype(np.float32)
    lookback = cfg.get('lookback', 50)
    return model, mean, scale, lookback

# -------------------------- single‑file inference ----------------------------

def predict_csv(model, mean, scale, lookback, csv_path):
    df = pd.read_csv(csv_path)
    assert {'log_time','RC_0','RC_1','voltage'}.issubset(df.columns), 'missing columns'

    # --- static branch (global scaler) ---
    static_raw  = df[['RC_0','RC_1']].values.astype(np.float32)
    static_norm = ((static_raw - mean) / scale).astype(np.float32)   # (N,2)
    print("static norm",static_norm)
    # --- dynamic branch: log_time z‑score per file ---
    t_raw = df['log_time'].values.astype(np.float32)
    t_norm= ((t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)).astype(np.float32)
    seq_feat = t_norm[:, None]                                        # (N,1)

    voltage = df['voltage'].values.astype(np.float32)
    N = len(df)
    X_seq, X_sta, Y, times = [], [], [], []
    for i in range(N - lookback):
        X_seq.append(seq_feat[i:i+lookback])
        X_sta.append(static_norm[i])
        Y.append(voltage[i+lookback])
        times.append(t_raw[i+lookback])

    X_seq = torch.tensor(np.stack(X_seq), dtype=torch.float32, device=DEVICE)
    X_sta = torch.tensor(np.stack(X_sta), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred = model(X_seq, X_sta).cpu().numpy()

    real = np.array(Y, dtype=np.float32)
    r2   = r2_score(real, pred)

    # plot
    order = np.argsort(times)
    plt.figure(figsize=(8,4))
    plt.plot(np.array(times)[order], real[order], label='True', linewidth=1)
    plt.plot(np.array(times)[order], pred[order], label='Pred', alpha=0.7)
    plt.xlabel('log_time'); plt.ylabel('Voltage')
    plt.title(f'{os.path.basename(csv_path)} | R²={r2:.4f}')
    plt.legend(); plt.grid(True); plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{os.path.basename(csv_path).replace('.csv','')}_pred.png")
    plt.close()
    return r2

# -------------------------- batch loop --------------------------------------

def main(data_dir, model_path, start_idx, end_idx):
    model, mean, scale, lookback = load_model(model_path)
    r2_list = []
    for idx in range(start_idx, end_idx+1):
        fp = os.path.join(data_dir, f'RC_cated_{idx}.csv')
        if not os.path.isfile(fp):
            print(f'[warn] {fp} missing'); continue
        print(f'Infer {os.path.basename(fp)} ...', end=' ')
        try:
            r2 = predict_csv(model, mean, scale, lookback, fp)
            r2_list.append(r2)
            print(f'R²={r2:.4f}')
        except Exception as e:
            print(f'Error: {e}')

    avg = np.mean(r2_list) if r2_list else float('nan')
    report = f'平均 R² (idx {start_idx}-{end_idx}): {avg:.4f}\n' + \
             '\n'.join([f'{i}: {v:.4f}' for i,v in zip(range(start_idx,start_idx+len(r2_list)), r2_list)])
    os.makedirs('results', exist_ok=True)
    with open('results/avg_r2.txt','w',encoding='utf8') as f: f.write(report)
    print('\n=== 结果 ===\n'+report+'\n写入 results/avg_r2.txt')

# -------------------------- run --------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./')
    ap.add_argument('--model', default='models/static_cond_model.pth')
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    args = ap.parse_args()
    main(args.data_dir, args.model, args.start, args.end)


