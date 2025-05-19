# ---------- utils_load.py ----------
import torch, numpy as np, os
from model import StaticCondTransformer      # 确保指向正确信息
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path: str,
               seq_dim: int,
               static_dim: int):
    """
    从 checkpoint 还原:
      • StaticCondTransformer
      • scaler μ / σ
      • lookback
    """
    ckpt = torch.load(path, map_location=DEVICE)

    cfg  = ckpt.get('cfg', {})
    model = StaticCondTransformer(seq_dim=seq_dim,
                                  static_dim=static_dim,
                                  d_model = cfg.get('d_model', 64),
                                  nhead   = cfg.get('heads', 4),
                                  num_layers = cfg.get('layers', 3)
                                 ).to(DEVICE)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state); model.eval()

    mean  = ckpt.get('scaler_mean', np.array([0.],dtype=np.float32)).astype(np.float32)
    scale = ckpt.get('scaler_scale', np.array([1.],dtype=np.float32)).astype(np.float32)
    lookback = cfg.get('lookback', 50)
    return model, mean, scale, lookback
