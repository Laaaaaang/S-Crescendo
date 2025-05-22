#!/bin/bash 
input_prefix="testsuite_"
output_prefix="result_"
input_extension=".sp"
output_extension=".lis"

for input_file in ${input_prefix}*${input_extension}; do
  
    file_number=$(echo "$input_file" | grep -oP "(?<=${input_prefix})\d+(?=${input_extension})")
    
    output_file="${output_prefix}${file_number}${output_extension}"
    
    echo "Running simulation for $input_file..."
    
  /home/kobe/eda/snps/hspice/U-2023.03-SP2-2/hspice/bin/hspice "$input_file" -o "./lis/$output_file"
    
    echo "Result saved to $output_file"

done
