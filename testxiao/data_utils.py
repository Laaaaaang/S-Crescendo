import torch
import numpy as np
from scipy.interpolate import make_interp_spline
def find_t(orig__volt, orig__t):
    orig__t, indices = torch.sort(orig__t)
    orig__volt = orig__volt[indices]

    __volt, __t = orig__volt, orig__t
    temp_sp = make_interp_spline(orig__t, orig__volt, k=1)
    fake_t = torch.linspace(orig__t[0], orig__t[-1], 1000)
    fake_v = torch.tensor(temp_sp(fake_t))
    index = torch.argmin((fake_v - 0.5).abs())

    # 边界保护
    left = max(index - 3, 0)
    right = min(index + 4, len(fake_t))
    __t = fake_t[left:right]
    __volt = fake_v[left:right]

    if len(__t) < 2:
        raise ValueError("在处理 find_t 时采样点过少，无法拟合。检查原始数据是否正常！")

    __volt, i = torch.sort(__volt, dim=0)
    __t = __t[i]
    print("Check point", __volt)
    spline = make_interp_spline(__volt, __t, k=1)
    res = spline(0.5)
    return res
