import pandas as pd
import re
import os

def convert_units(value):
    """¸ù¾Ýµ¥Î»Ç°×º½øÐÐ×ª»»"""
    if value is None or (isinstance(value, str) and value.strip() == ''):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    match = re.match(r'([\d\.Ee+-]+)([fpnumkMG]?)$', str(value), re.IGNORECASE)
    if not match:
        return None

    num_part, unit = match.groups()
    unit = unit.lower()

    try:
        num = float(num_part)
    except ValueError:
        return None

    unit_conversions = {
        'u': 1e-6, 'm': 1e-3, 'f': 1e-15,
        'n': 1e-9, 'p': 1e-12, 'k': 1e3,
        'M': 1e6, 'G': 1e9
    }
    return num * unit_conversions.get(unit, 1)

def process_spice_file(input_file, output_csv):
    """´¦Àíµ¥¸öSPICEÎÄ¼þ"""
    # ¶ÁÈ¡ÎÄ¼þÄÚÈÝ
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # ¶¨ÒåÄ¿±êÐÐ±êÌâ¹Ø¼ü´Ê
    header_keywords = ["time", "voltage"]

    # ÕÒµ½°üº¬Ä¿±êÁÐ±êÌâµÄÐÐºÅ
    header_line_index = None
    for i, line in enumerate(lines):
        if all(keyword in line.lower() for keyword in header_keywords):
            header_line_index = i
            break

    if header_line_index is None:
        print(f"Î´ÕÒµ½Ä¿±êÊý¾ÝµÄÆðÊ¼ÐÐ: {input_file}")
        return

    # ÌáÈ¡ÓÐÐ§Êý¾ÝÐÐ
    time_data = []
    voltage_data = []
    for line in lines[header_line_index + 1:]:
        line = line.strip()
        if line and not line.startswith('*****'):
            parts = line.split()
            if len(parts) >= 2:
                # ¼ì²éÊý¾ÝÊÇ·ñÓÐÐ§£¨·Ç¿Õ×Ö·û´®£©
                if parts[0].strip() and parts[1].strip():
                    time_data.append(parts[0])
                    voltage_data.append(parts[1])

    # ×ª»»µ¥Î»²¢¹ýÂËNoneÖµ
    time_data = [t for t in [convert_units(t) for t in time_data] if t is not None]
    voltage_data = [v for v in [convert_units(v) for v in voltage_data] if v is not None]

    # È·±£Êý¾Ý³¤¶ÈÒ»ÖÂ
    min_length = min(len(time_data), len(voltage_data))
    time_data = time_data[:min_length]
    voltage_data = voltage_data[:min_length]

    # ´´½¨DataFrame
    df = pd.DataFrame({
        "Time": time_data,
        "Voltage": voltage_data
    })

    # È¥³ý¿ÕÐÐ
    df = df.dropna()

    # Ð´ÈëCSVÎÄ¼þ
    df.to_csv(output_csv, index=False)
    print(f"Êý¾ÝÒÑ³É¹¦Ð´Èë {output_csv} (¹² {len(df)} ÐÐÓÐÐ§Êý¾Ý)")

def process_multiple_lis_files(n):
    """ÅúÁ¿´¦ÀílisÎÄ¼þ"""
    # È·±£Êä³öÄ¿Â¼´æÔÚ
    os.makedirs("./result", exist_ok=True)
    
    for i in range(1, n + 1):
        input_file = f"result_{i}.lis"
        output_csv = f"./result/result_{i}.csv"
        
        if os.path.exists(input_file):
            process_spice_file(input_file, output_csv)
        else:
            print(f"ÎÄ¼þ²»´æÔÚ: {input_file}")

# ´¦Àí5¸öÎÄ¼þ
process_multiple_lis_files(501)