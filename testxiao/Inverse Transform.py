import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def inverse_log_transform(log_time_tensor, mid_t_tensor):
    """
    log_time_tensor: torch.tensor, 已变换的 log_time
    mid_t_tensor: torch.tensor, 还原用的中心点 (find_t 得到的)
    """
    sign = torch.sign(log_time_tensor)
    sign[sign == 0] = 1
    abs_t = (torch.exp(log_time_tensor.abs() + torch.log(torch.tensor(0.1))) - 0.1) / 1e10
    restored_t = mid_t_tensor + sign * abs_t
    return restored_t

def restore_csv(
    file_path,    # 输入变换后的csv路径
    save_path,    # 输出还原后的csv路径
    mid_t,
    vdd=1.1       # 电压归一化还原需要知道VDD  
):
    df = pd.read_csv(file_path)

    if 'log_time' not in df.columns or 'real' not in df.columns:
        raise ValueError(f"文件 {file_path} 缺少 'log_time' 或 'voltage' 列")

    log_time = torch.tensor(df['log_time'].values, dtype=torch.float32)
    voltage = torch.tensor(df['pred'].values, dtype=torch.float32)

    # # 重新估计中心 t_mid（从数据中找）
    # from data_utils import find_t
    # fake_t = torch.linspace(log_time.min(), log_time.max(), 1000)

    # fake_v_np = np.interp(fake_t.numpy(), log_time.numpy(), voltage.numpy())
    # fake_v = torch.tensor(fake_v_np, dtype=torch.float32)

    # mid_t = torch.tensor(find_t(fake_v, fake_t), dtype=torch.float32)   # ✅ 这里保证是 torch tensor
    mid_t = 8.836961061036981e-10
    restored_t = inverse_log_transform(log_time, mid_t)
    restored_voltage = voltage * vdd  # 恢复实际电压

    df_restored = pd.DataFrame({
        'time': restored_t.numpy(),
        'voltage': restored_voltage.numpy()
    })

    df_restored.to_csv(save_path, index=False)
    print(f"✅ 还原成功保存: {save_path}")

    # 画恢复后的波形图
    plt.figure(figsize=(10, 5))
    plt.plot(restored_t.numpy(), restored_voltage.numpy(), color="blue", label="Restored Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Restored Time-Domain Waveform")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_save_path = save_path.replace(".csv", "_restored.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"🖼️ 恢复波形图保存: {plot_save_path}")


def batch_restore(
    base_dir,     # 输入目录
    save_dir,     # 输出目录
    file_list,    # 要处理的文件列表
    mid_t,
    vdd=1.1       # 电压值
):
    os.makedirs(save_dir, exist_ok=True)
    for fname in file_list:
        restore_csv(
            file_path=os.path.join(base_dir, fname),
            save_path=os.path.join(save_dir, fname.replace('.csv', '_restored.csv')),
            mid_t=mid_t,
            vdd=vdd
        )

# ✅ 示例入口
if __name__ == "__main__":
    batch_restore(
        base_dir="testxiao/inverse",
        save_dir="testxiao/result/Inverse_Transform",
        file_list=["RC_cated_181_2.csv"],
        mid_t = 1.9973460714170096e-09,
        vdd=1.1
    )
