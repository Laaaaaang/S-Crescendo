import pandas as pd
import time
import os

suffixes = range(1,501)

print(f"µ±Ç°Â·¾¶: {os.getcwd()}")

start_time = time.time()

for i in suffixes:
    start_time1 = time.time()

    file1_name = f'RC/testsuite_{i}.csv'
    file2_name = f'lis/result/result_{i}.csv'
    output_file_name = f'result/RC_cated_{i}.csv'

    # ¶ÁÈ¡RCÎÄ¼þ£¬Ö»¶ÁÈ¡µÚÒ»ÐÐ
    try:
        file1 = pd.read_csv(file1_name, header=None)
        data_to_copy = file1.iloc[0].values
    except Exception as e:
        print(f"¶ÁÈ¡ {file1_name} Ê§°Ü: {e}")
        continue

    # ¶ÁÈ¡´¦Àí½á¹ûÎÄ¼þ
    try:
        file2 = pd.read_csv(file2_name)
    except Exception as e:
        print(f"¶ÁÈ¡ {file2_name} Ê§°Ü: {e}")
        continue

    # Ö±½ÓÆ´½ÓÊý¾Ý
    repeated_data = pd.DataFrame([data_to_copy] * len(file2), 
                               columns=[f"RC_{x}" for x in range(len(data_to_copy))])
    
    # ºáÏòÆ´½ÓÁ½¸öDataFrame
    result = pd.concat([file2, repeated_data], axis=1)

    # ±£´æÎÄ¼þ
    try:
        result.to_csv(output_file_name, index=False)
        end_time1 = time.time()
        print(f'ÎÄ¼þ{i}´¦ÀíÍê³É£¬ÓÃÊ±: {end_time1 - start_time1:.2f} Ãë')
    except Exception as e:
        print(f"±£´æ {output_file_name} Ê§°Ü: {e}")

end_time = time.time()
print(f'×ÜÓÃÊ±: {end_time - start_time:.2f} Ãë')