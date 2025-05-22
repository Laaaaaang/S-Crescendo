import re
import csv
import os
import numpy as np
from glob import glob
from scipy.signal import ss2tf, residue

def parse_spice_file(file_path):
    """解析 SPICE 文件中的 R1,R2,C1,C2 参数"""
    values = {'R1': None, 'R2': None, 'C1': None, 'C2': None}
    unit_scale = {'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'μ':1e-6,
                  'm':1e-3,'k':1e3,'meg':1e6,'g':1e9}
    def convert(v):
        s = v.lower().replace(' ', '')
        if 'e' in s:
            return float(s)
        m = re.match(r'^([\d.]+)([a-zμ]*)$', s)
        if not m:
            raise ValueError(f"Invalid value '{v}'")
        num, unit = m.groups()
        return float(num) * unit_scale.get(unit, 1.0)

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            parts = re.split(r'\s+', line)
            if len(parts)>=4 and parts[0] in values:
                values[parts[0]] = convert(parts[3])

    missing = [k for k,v in values.items() if v is None]
    if missing:
        raise ValueError(f"Missing components: {missing}")
    R = [values['R1'], values['R2']]
    C = [values['C1'], values['C2']]
    return R, C

def build_state_space(R, C):
    """构建 2 阶 RC 串联网络的状态空间模型"""
    R = np.array(R, float)
    C = np.array(C, float)
    # 导纳矩阵 G
    G = np.array([[1/R[0]+1/R[1], -1/R[1]],
                  [-1/R[1],         1/R[1]]], float)
    Cmat = np.diag(C)
    B = np.array([1/R[0], 0], float)
    A = -np.linalg.solve(Cmat, G)
    B = np.linalg.solve(Cmat, B)
    C_vec = np.array([0,1], float)
    return A, B, C_vec

def compute_A_and_poles(R, C):
    """
    计算 A' 和极点 p：
      1) 用 residue 得到 r_i, p_i
      2) 令 A_i' = τ_i * r_i = -r_i / p_i
    保证 ∑ A_i' = 1
    """
    A, B, C_vec = build_state_space(R, C)
    num_mat, den = ss2tf(A, B.reshape(-1,1), C_vec, 0)
    num = np.ravel(num_mat)
    # 如果分子阶≥分母阶，去掉最高多项式项
    if len(num) >= len(den):
        num = num[-(len(den)-1):]
    # 部分分式展开
    r, p, _ = residue(num, den)
    # 过滤虚部噪声
    mask = np.abs(np.imag(p)) < 1e-6
    r = np.real(r[mask])
    p = np.real(p[mask])
    # 排序
    idx = np.argsort(np.abs(p))
    r = r[idx]
    p = p[idx]
    # 计算 A' = -r/p
    A_pr = -r / p
    # 归一化（可选，防止微小数值误差）
    A_pr = A_pr / np.sum(A_pr)
    return A_pr, p

def save_results(A_pr, poles, filename):
    """一行写入 A1,p1,A2,p2"""
    row = []
    for a, p in zip(A_pr, poles):
        row.extend([a, p])
    with open(filename, 'w', newline='') as f:
        csv.writer(f).writerow(row)

def validate_poles(poles):
    """检查极点均为负实数"""
    if np.any(poles >= 0):
        raise ValueError("Non-negative pole detected")

if __name__ == '__main__':
    input_dir  = 'sp_files_2'
    output_dir = 'result_2'
    os.makedirs(output_dir, exist_ok=True)

    for sp in glob(os.path.join(input_dir, '*.sp')):
        print("Processing", sp)
        try:
            R, C = parse_spice_file(sp)
            print("  R:", R, "C:", C)
            A_pr, poles = compute_A_and_poles(R, C)
            validate_poles(poles)

            base = os.path.splitext(os.path.basename(sp))[0]
            out_csv = os.path.join(output_dir, base + '.csv')
            save_results(A_pr, poles, out_csv)

            print("  A':", [f"{float(a):.3f}" for a in A_pr])
            print("  Poles:", [f"{float(p):.2e}" for p in poles])
            print("  sum A':", sum(A_pr))
            print("  Saved to", out_csv)
        except Exception as e:
            print("  Error:", e)
    print("Done.")

