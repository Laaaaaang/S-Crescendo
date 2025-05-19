import torch
# from Main.defaults import db_raw_train_path
from data_utils import find_t
import matplotlib.pyplot as plt
lst = torch.load('../0_raw_data/db20241105_train_raw.pth')


for line in lst:
    condition, seq, primary_key = line
    condition_continuous, condition_discrete = condition
    vdd=condition_continuous[2]
    if seq[:,0].max()<1e-10:
        volt=seq[:,1]/vdd
        ttt = find_t(volt, seq[:, 0])
        new_t = seq[:, 0] - ttt
        sign = torch.sign(new_t)
        sign[sign == 0] = 1
        new_t = (torch.log(new_t.abs() * 1e10 + 0.1)-torch.log(torch.tensor(0.1))) * sign
        plt.plot(new_t, volt)

plt.show()
#/3 -2.05 -2.3
#-6.1 6.87
from data_inverse_db import Mapping
a=Mapping()
b=torch.linspace(0,1e-9,100000000)
mid_t=torch.tensor([1e-11])
xt=a.time_direct(b,mid_t)
print(xt.max())
et=a.time_inverse(xt,mid_t)
et=(et-b).abs()
print(et.max())
print(et.mean())