import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
import os

def analyze_fit(real_file, pred_file, save_dir="analysis_result"):
    # 读取真实值和预测值
    real_df = pd.read_csv(real_file)
    pred_df = pd.read_csv(pred_file)

    # 自动对齐时间列
    if "time" in real_df.columns:
        time_real = real_df["time"].values
    else:
        raise ValueError("real.csv缺少'time'列")

    if "time" in pred_df.columns:
        time_pred = pred_df["time"].values
    else:
        raise ValueError("pred_inverse.csv缺少'time'列")

    # 自动同步长度
    min_len = min(len(time_real), len(time_pred))
    time_real = time_real[:min_len]
    time_pred = time_pred[:min_len]

    # 取真实电压和预测电压
    if "voltage" in real_df.columns:
        real_voltage = real_df["voltage"].values
    elif "true_voltage" in real_df.columns:
        real_voltage = real_df["true_voltage"].values
    else:
        raise ValueError("real.csv缺少'voltage'或'true_voltage'列")

    if "mask_pred" in pred_df.columns:
        pred_voltage = pred_df["mask_pred"].values
    elif "voltage" in pred_df.columns:
        pred_voltage = pred_df["voltage"].values
    else:
        raise ValueError("pred_inverse.csv缺少'mask_pred'或'voltage'列")

    # 同步裁剪电压数据
    real_voltage = real_voltage[:min_len]
    pred_voltage = pred_voltage[:min_len]

    # 计算指标
    r2 = r2_score(real_voltage, pred_voltage)
    rmse = np.sqrt(mean_squared_error(real_voltage, pred_voltage))
    mae = mean_absolute_error(real_voltage, pred_voltage)
    max_err = max_error(real_voltage, pred_voltage)

    print(f"R2 Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"Max Error: {max_err:.6e}")

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/fit_metrics.txt", "w") as f:
        f.write(f"R2 Score: {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.6e}\n")
        f.write(f"MAE: {mae:.6e}\n")
        f.write(f"Max Error: {max_err:.6e}\n")
    print(f"✅ 拟合指标保存到 {save_dir}/fit_metrics.txt")

    # 保存预测 vs 真实对比图
    plt.figure(figsize=(10, 5))
    plt.plot(time_real, real_voltage, label="True Voltage", linewidth=2)
    plt.plot(time_real, pred_voltage, label="Predicted Voltage", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Predicted vs True Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/pred_vs_true.png")
    plt.close()
    print(f"✅ 预测与真实对比图保存到 {save_dir}/pred_vs_true.png")

    # 保存误差随时间变化图
    error = pred_voltage - real_voltage

    plt.figure(figsize=(10, 5))
    plt.plot(time_real, error, label="Prediction Error", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (V)")
    plt.title("Prediction Error Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_vs_time.png")
    plt.close()
    print(f"✅ 误差曲线保存到 {save_dir}/error_vs_time.png")

    # 保存误差直方图
    plt.figure(figsize=(8, 5))
    plt.hist(error, bins=100, color="blue", alpha=0.7)
    plt.xlabel("Error (V)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_histogram.png")
    plt.close()
    print(f"✅ 误差直方图保存到 {save_dir}/error_histogram.png")


if __name__ == "__main__":
    analyze_fit(
        real_file="testxiao/result/Inverse_Transform/real.csv",
        pred_file="testxiao/result/Inverse_Transform/pred_inverse.csv",
        save_dir="analysis_result"
    )
