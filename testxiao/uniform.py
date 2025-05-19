# ========================= uniform.py =========================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def uniform_interpolation(t_raw, v_raw, total_samples=1000):
    """仅按时间均匀插值"""
    interp_t = np.linspace(t_raw.min(), t_raw.max(), total_samples)
    interp_v = interp1d(t_raw, v_raw, kind='linear', fill_value="extrapolate")(interp_t)
    return interp_t, interp_v


def voltage_interpolation_uniform(
        file_path, save_path,
        time_col="log_time", voltage_col="voltage",
        total_samples=1000
):
    df = pd.read_csv(file_path)
    t_raw, v_raw = df[time_col].values, df[voltage_col].values
    feature_cols = [c for c in df.columns if c not in (time_col, voltage_col)]
    static_feat = df[feature_cols].iloc[-1].values

    # 1) 时间排序  2) 均匀插值
    idx = np.argsort(t_raw)
    new_t, new_v = uniform_interpolation(t_raw[idx], v_raw[idx], total_samples)

    # 保存
    repeated_feat = np.tile(static_feat, (total_samples, 1))
    out_df = pd.concat([
        pd.DataFrame({"log_time": new_t, "voltage": new_v}),
        pd.DataFrame(repeated_feat, columns=feature_cols)
    ], axis=1)
    out_df.to_csv(save_path, index=False)
    print(f"✅ 均匀插值: {save_path}")

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.scatter(t_raw, v_raw, s=10, label="Original")
    plt.scatter(new_t, new_v, s=2, label="Uniform")
    plt.xlabel("Log‑Time"); plt.ylabel("Voltage"); plt.title(os.path.basename(file_path))
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path.replace(".csv", "_scatter.png")); plt.close()


def batch_voltage_interpolation_uniform(
        base_dir, save_dir,
        file_prefix="RC_cated_", file_suffix=".csv",
        file_list=None, total_samples=1000
):
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 自动扫描 ----------
    if file_list is None:
        pattern = os.path.join(base_dir, f"{file_prefix}*{file_suffix}")
        file_list = [os.path.basename(p) for p in glob.glob(pattern)]
        file_list.sort()

    for fname in file_list:
        voltage_interpolation_uniform(
            file_path=os.path.join(base_dir, fname),
            save_path=os.path.join(save_dir, fname),
            total_samples=total_samples
        )


if __name__ == "__main__":
    # 默认处理 1‑100 号文件；若需全部自动扫描，file_list=None
    file_list = [f"RC_cated_{i}.csv" for i in range(1, 501)]
    batch_voltage_interpolation_uniform(
        base_dir="testxiao/result",
        save_dir="testxiao/result/final_uniform",
        file_list=file_list,      # 或设为 None 自动搜索
        total_samples=1000
    )
