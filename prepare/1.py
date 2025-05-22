import os

def generate_spice_file(n, r_value, c_value, output_dir='sp_files'):
    """生成SPICE文件到指定目录"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
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

*—— 并联 RC ——*
* R1 从 OUTPUT 到地
R1 OUT OUTPUT {r_value}

C1 OUTPUT 0 {c_value}

.END
"""
    with open(file_name, 'w') as file:
        file.write(spice_content)
    print(f"SPICE file '{file_name}' 已生成。")

# 批量生成100个测试文件
for n in range(1,501):  # 生成1-100
    r = 10 + 10 * n             # 电阻值 (Ω)
    c = 10e-15 + 3e-15 * n       # 电容值 (F)
    c_ff = c * 1e15              # 转换为fF
    c_str = f"{c_ff:.8f}f".replace(".00000000", "")  # 优化格式
    
    generate_spice_file(n, r, c_str)

print("\n所有SPICE文件已生成到 sp_files 目录")

