import os
import pandas as pd
import torch
import numpy as np
from data_utils import find_t
import traceback
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import find_t

def process_csv_files(save_dir: str, base_dir: str, file_prefix: str = "RC_cated_", file_suffix: str = ".csv", file_indices: list = None, save_suffix=".csv"):
    """
    对一系列 RC_cated_i.csv 文件进行时序对数中心化变换，保存新 CSV，同时绘制变换前后的波形对比图。
    """
    if file_indices is None:
        raise ValueError("请提供要处理的文件索引列表")

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for i in file_indices:
        file_path = os.path.join(base_dir, f"{file_prefix}{i}{file_suffix}")
        if not os.path.exists(file_path):
            print(f"跳过: {file_path} 不存在")
            continue
        
        # 读取数据
        df = pd.read_csv(file_path)
        if df.shape[1] < 2:
            print(f"文件 {file_path} 列数不足2列")
            continue

        t_col = df.columns[0]
        v_col = df.columns[1]
        t_raw = torch.tensor(df[t_col].values, dtype=torch.float32)
        v_raw = torch.tensor(df[v_col].values, dtype=torch.float32)
        
        # 归一化电压
        vdd = 1.1  # 可修改为实际值
        volt = v_raw / vdd
        # print(f"volt: {volt}")
        # print(f"t_raw: {t_raw}")

        try:
            # 查找中心点并做时间变换
            ttt = find_t(volt, t_raw)
            print("t_mid",ttt)
            new_t = t_raw - ttt
            sign = torch.sign(new_t)
            sign[sign == 0] = 1
            new_t = (torch.log(new_t.abs() * 1e10 + 0.1) - torch.log(torch.tensor(0.1))) * sign

            rest_features = df.iloc[:, 2:]
            transformed_df = pd.concat([
                pd.DataFrame({'log_time': new_t.numpy(), 'voltage': volt.numpy()}),
                rest_features.reset_index(drop=True)
            ], axis=1)

            save_path = os.path.join(save_dir, f"{file_prefix}{i}{save_suffix}")
            transformed_df.to_csv(save_path, index=False)
            print(f"✅ 保存成功: {save_path}")

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(t_raw.numpy(), volt.numpy(), label='Original', color='blue')
            plt.xlabel("Time")
            plt.ylabel("Voltage")
            plt.title(f"RC_{i} raw")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(new_t.numpy(), volt.numpy(), label='Log-Centered', color='orange')
            plt.xlabel("Log-Transformed Time")
            plt.ylabel("Voltage")
            plt.title(f"RC_{i} after log")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"RC_{i}_compare.png"))
            plt.close()
            print(f"🖼️ 绘图完成: RC_{i}_compare.png")

        except Exception as e:
            print(f"❌ 处理文件 {file_path} 时出错: {e}")
            traceback.print_exc()

# === 批量入口（覆盖原文件最末尾那段即可） ===
if __name__ == "__main__":
    process_csv_files(
        save_dir="testxiao/result",
        base_dir="testxiao/basic/model6",
        # 这里直接一次性处理 1‑100 号文件
        file_indices=list(range(181, 182))     # [1, 2, …, 100]
    )

