# -*- coding: utf-8 -*-
"""
pipeline_train_error_models.py
==============================
从 2 阶到 N 阶自动化生成  error_n  数据集并逐阶训练模型。

目录结构假定：
```
project/
├── model1.pth                 # 已训练好的一阶模型
├── order2/  *.csv             # 2 阶真实文件夹  (列: log_time, voltage, RC_0..RC_3)
├── order3/  *.csv             # 3 阶真实文件夹  (列到 RC_5)
├── order4/  *.csv             # ...
└── orderN/  *.csv
```
脚本将：
1. 按 2 → N 递增：
   * 使用上一阶 error 模型作为 teacher 递归生成  `error_k_dataset.csv`。
   * 调用 `train_error_model()` 训练  `model_error_k.pth`，保存在 `error_models/`。
2. 完成后，可直接用这些模型级联推理。

> 需先放置可用的一阶模型 `model1.pth`。
> `train_error_model()` 与 `build_dataset_recursive()` 已在 canvas 中， 这里直接 `import`。
"""

import os, argparse, glob, importlib.util, sys, shutil

# ---------- 动态 import 之前写入的两个脚本 ----------
_this_dir = os.path.dirname(__file__)

for mod_name, file_name in [
        ('build_dataset_recursive', 'build_dataset_recursive.py'),
        ('train_error_model',       'train_error_model.py')]:
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_this_dir, file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

from build_dataset_recursive import build_dataset_recursive
from train_error_model     import train

# ---------- helper ----------

def concatenate_datasets(tmp_folder, order_k):
    files = sorted(glob.glob(os.path.join(tmp_folder, f'order{order_k}_*.csv')))
    if not files: raise RuntimeError('no temp dataset found')
    dfs = [__import__('pandas').read_csv(f) for f in files]
    merged = __import__('pandas').concat(dfs, ignore_index=True)
    final_path = os.path.join(tmp_folder, f'error{order_k}_dataset.csv')
    merged.to_csv(final_path, index=False)
    # 清理单文件
    for f in files: os.remove(f)
    return final_path

# ---------- pipeline ----------

def pipeline(base_dir, model1_path, start=3, end=5, lookback=50, epochs=50):
    os.makedirs('error_models', exist_ok=True)
    tmp_ds_dir = os.path.join(base_dir, '_tmp_ds')
    os.makedirs(tmp_ds_dir, exist_ok=True)

    for k in range(start, end+1):
        print("k",k)
        order_dir = os.path.join('data', f'order{k}')
        if not os.path.isdir(order_dir):
            print(f'[skip] folder {order_dir} missing'); break

        print(f'\n### Building dataset for order {k}')
        # 逐文件生成 error_k 临时 csv
        for fp in glob.glob(os.path.join(order_dir, '*.csv')):
            fname = os.path.splitext(os.path.basename(fp))[0]
            out_csv = os.path.join(tmp_ds_dir, f'order{k}_{fname}.csv')
            build_dataset_recursive(fp, model1_path, 'error_models', out_csv)

        merged_csv = concatenate_datasets(tmp_ds_dir, k)
        print(f' merged dataset → {merged_csv}')

        print(f'### Training error_{k} model')
        model_out = os.path.join('error_models', f'model_error_{k}.pth')
        train(merged_csv, epochs=epochs, lookback=lookback, model_out=model_out)

    # shutil.rmtree(tmp_ds_dir)
    print('\n✅ Pipeline finished. All models saved in error_models/')

# ---------- CLI ----------
if __name__ == '__main__':
    ap = argparse.ArgumentParser('Recursive error model training pipeline')
    ap.add_argument('--base_dir', default='.', help='project base folder')
    ap.add_argument('--model1',   default='models/static_cond_model.pth')
    ap.add_argument('--min_order', type=int, default=2)
    ap.add_argument('--max_order', type=int, default=5)
    ap.add_argument('--lookback', type=int, default=20)
    ap.add_argument('--epochs',   type=int, default=40)
    args = ap.parse_args()

    pipeline(args.base_dir, args.model1, args.min_order, args.max_order, args.lookback, args.epochs)

# python pipeline_train_error_models.py --base_dir . --model1    models/static_cond_model.pth --min_order 3 --max_order 5 --lookback 20 --epochs    60
