import re
import csv
import os
from glob import glob

def parse_spice_file(file_path):
    """解析 SPICE 网表，提取 R1 和 C1 的值"""
    r_value = None
    c_value = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 匹配 R1 行（如 "R1 OUT OUTPUT 180"）
            if line.startswith('R1'):
                parts = re.split(r'\s+', line)
                r_value = float(parts[3])  # 提取电阻值（单位：Ω）
            
            # 匹配 C1 行（如 "C1 OUTPUT 0 90f"）
            elif line.startswith('C1'):
                parts = re.split(r'\s+', line)
                cap_value = parts[3].lower()
                
                # 处理单位（fF、pF 等）
                if 'f' in cap_value:  # 飞法（femtofarad）
                    c_value = float(cap_value.replace('f', '')) * 1e-15
                elif 'p' in cap_value:  # 皮法（picofarad）
                    c_value = float(cap_value.replace('p', '')) * 1e-12
                else:  # 默认单位：法拉（无单位）
                    c_value = float(cap_value)
    
    return r_value, c_value

def calculate_pole(r, c):
    """计算极点：pole = -1 / (R * C)"""
    if r is None or c is None:
        raise ValueError("R1 或 C1 未在网表中定义！")
    return -1 / (r * c)

def save_to_csv(pole, output_file):
    """将极点和分子系数保存到 CSV 文件"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([1.0, pole])

if __name__ == '__main__':
    input_dir = 'sp_files'  # SPICE文件目录
    os.makedirs(input_dir, exist_ok=True)  # 确保目录存在
    
    # 获取目录下所有.sp文件
    sp_files = glob(os.path.join(input_dir, '*.sp'))
    
    if not sp_files:
        print(f"错误：目录 {input_dir} 中没有找到.sp文件！")
        exit()
    
    for sp_file in sp_files:
        try:
            # 解析文件
            r, c = parse_spice_file(sp_file)
            
            # 计算极点
            pole = calculate_pole(r, c)
            
            # 生成对应的CSV文件名（如 testsuite_1.sp -> testsuite_1.csv）
            csv_file = os.path.splitext(sp_file)[0] + '.csv'
            
            # 保存结果
            save_to_csv(pole, csv_file)
            print(f"处理完成: {os.path.basename(sp_file)} -> {os.path.basename(csv_file)} | Pole = {pole:.2e} rad/s")
            
        except Exception as e:
            print(f"处理失败: {os.path.basename(sp_file)} | 错误: {str(e)}")
            continue

    print("\n所有文件处理完成，每个.sp文件已生成对应的.csv文件")