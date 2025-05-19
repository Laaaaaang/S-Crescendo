# -*- coding: utf-8 -*-
"""
build_model2_dataset.py  ▸  根据二阶数据文件生成『模型 2』训练集 (升级版)
================================================================
输入文件格式
--------------
每个 *二阶* CSV 含列：

| 列名      | 含义                       |
|-----------|---------------------------|
| `log_time`| 时间序列 *t*              |
| `voltage` | 二阶真实响应 *y_true(t)*  |
| `RC_0`    | **第 1 分式** 系数 A₁     |
| `RC_1`    | **第 1 分式** 极点 p₁     |
| `RC_2`    | **第 2 分式** 系数 A₂     |
| `RC_3`    | **第 2 分式** 极点 p₂     |

步骤逻辑
--------
1. **模型 1 推理**
   * 用 *(A₁, p₁)* 预测一阶响应 `S1_raw`；再用 *(A₂, p₂)* 预测 `S2_raw`。
   * 最终结果按系数比例缩放：
     ```python
     S1 = A₁ / (A₁ + A₂) * (S1_raw + S2_raw)   # 保证能量按系数分配
     S2 = A₂ / (A₁ + A₂) * (S1_raw + S2_raw)
     ```
     > 若模型 1 已内部乘以系数，可把比例逻辑删掉。

2. `M1 = S1 + S2`
3. `error = y_true – M1`
4. 对序列中每个时间点写一行：
   `error, t, 2, A₁, p₁, A₂, p₂`

输出
----
生成单个汇总 CSV（默认 `model2_dataset.csv`）。

运行示例
---------
```bash
python build_model2_dataset.py --data_dir  data/second_order --model1    models/model1.pth --out_csv   model2_dataset.csv
```
"""

import os, glob, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn

def load_model1(path: str):
    """加载一阶模型；请替换为真实实现."""
    class Dummy(nn.Module):
        def forward(self, pole: torch.Tensor):
            # 输入 pole (batch,1) → 输出随机形状 (batch,T)
            T = 500
            return torch.rand((pole.size(0), T), dtype=torch.float32)
    model = Dummy()
    # model.load_state_dict(torch.load(path))
    model.eval()
    return model

# ---------- 一阶预测 (示例) ----------

def predict_first_order(model1, pole_value: float, T: int):
    pole_tensor = torch.tensor([[pole_value]], dtype=torch.float32)
    with torch.no_grad():
        out = model1(pole_tensor)  # (1,T)
    # 若模型输出长度与真实序列不同，需插值 / 截断；此处直接 tile / 截断
    pred = out.squeeze(0).cpu().numpy()
    if len(pred) < T:  # pad
        pred = np.pad(pred, (0, T-len(pred)), mode='edge')
    return pred[:T].astype(np.float32)

# ---------- 解析二阶 csv ----------

def parse_csv(fp: str):
    df = pd.read_csv(fp)
    assert {'log_time','voltage','RC_0','RC_1','RC_2','RC_3'}.issubset(df.columns), '列不全'
    t  = df['log_time'].values.astype(np.float32)
    y  = df['voltage' ].values.astype(np.float32)
    A1,p1 = df['RC_0'][0], df['RC_1'][0]
    A2,p2 = df['RC_2'][0], df['RC_3'][0]
    return t,y, A1,p1, A2,p2

# ---------- 主构建函数 ----------

def build_dataset(data_dir: str, model1_path: str, out_csv: str):
    model1 = load_model1(model1_path)
    rows   = []

    for fp in sorted(glob.glob(os.path.join(data_dir,'*.csv'))):
        t, y_true, A1,p1, A2,p2 = parse_csv(fp)
        T = len(t)
        # 1) 用模型 1 预测（仅极点输入示例）
        S1_raw = predict_first_order(model1, p1, T)
        S2_raw = predict_first_order(model1, p2, T)

        # 系数比例缩放（如模型1已含系数可去掉）
        coef_sum = A1 + A2 if (A1+A2)!=0 else 1.0
        S1 = (A1/coef_sum) * (S1_raw + S2_raw)
        S2 = (A2/coef_sum) * (S1_raw + S2_raw)

        # 2) 叠加
        M1 = S1 + S2
        # 3) 误差
        err = y_true - M1

        # 4) 写每个时间点
        for e, tt in zip(err, t):
            rows.append([e, tt, 2, A1, p1, A2, p2])

    df_out = pd.DataFrame(rows, columns=['error','t','dim_ext','A1','p1','A2','p2'])
    df_out.to_csv(out_csv, index=False)
    print(f'Saved {len(df_out)} rows → {out_csv}')

# ---------- CLI ----------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='folder with 2‑order csvs')
    ap.add_argument('--model1', required=True, help='path to model1 .pth')
    ap.add_argument('--out_csv', default='model2_dataset.csv')
    args = ap.parse_args()
    build_dataset(args.data_dir, args.model1, args.out_csv)
