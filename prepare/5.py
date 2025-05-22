import os
import random
from itertools import combinations, permutations

# 输出目录
output_dir = "sp_files_5"
os.makedirs(output_dir, exist_ok=True)

def generate_spice_file(n, rc_tuple):
    """根据五元组 rc_tuple 生成 5 阶 RC 的 SPICE 文件"""
    # 解包 5 组 (R, C)
    r1, c1 = rc_tuple[0]
    r2, c2 = rc_tuple[1]
    r3, c3 = rc_tuple[2]
    r4, c4 = rc_tuple[3]
    r5, c5 = rc_tuple[4]

    # 格式化电容（以 fF 单位）
    c1_str = f"{c1 * 1e15:.8f}f"
    c2_str = f"{c2 * 1e15:.8f}f"
    c3_str = f"{c3 * 1e15:.8f}f"
    c4_str = f"{c4 * 1e15:.8f}f"
    c5_str = f"{c5 * 1e15:.8f}f"

    file_name = os.path.join(output_dir, f'testsuite_{n}.sp')
    spice_content = f"""
.global vdd
vdd vdd 0 1.1
.TRAN 10ps 20ns
.IC V(I1)=0 V(OUTPUT)=0

.PRINT  V(OUTPUT) I(R1) I(C1)
.INC "\\edatool\\40nm\\scc40nll_vhsc50_rvt.cdl"
.LIB "\\edatool\\40nm\\smic40llrf_1125_oa_cds_v1.15_2\\l0040ll_v1p15.lib" tt
X1 I1 OUT vdd 0 CLKINV3_12TR50

R1 OUT node1  {r1}
C1 node1 0   {c1_str}

R2 node1 node2  {r2}
C2 node2 0   {c2_str}

R3 node2 node3  {r3}
C3 node3 0   {c3_str}

R4 node3 node4  {r4}
C4 node4 0   {c4_str}

R5 node4 OUTPUT  {r5}
C5 OUTPUT 0   {c5_str}

.END
"""
    with open(file_name, 'w') as f:
        f.write(spice_content)
    print(f"Generated: {file_name}")

# 构造基础 RC 值列表（这里以 6 组为例，也可扩展更多）
base_rc = [(100 + 60 * i, 50e-15 + 30e-15 * i) for i in range(6)]

# 1) 生成所有 5 元组合（无序） C(6,5)=6 种
quintuples = list(combinations(base_rc, 5))

# 2) 对每个组合生成所有排列（有序） 6*5! = 720 种
ordered_quintuples = []
for quad in quintuples:
    ordered_quintuples.extend(permutations(quad, 5))

# 3) 随机打乱并取前 200 组（可调整为任意数量）
random.shuffle(ordered_quintuples)
selected_quintuples = ordered_quintuples[:200]

# 4) 批量生成 SPICE 文件
for idx, rc_tuple in enumerate(selected_quintuples, start=1):
    generate_spice_file(idx, rc_tuple)
