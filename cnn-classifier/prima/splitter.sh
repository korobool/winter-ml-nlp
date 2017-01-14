#! /bin/bash
# Configuration stuff

fspec=$1
num_files=$2

# Work out lines per file.

total_lines=$(wc -l <${fspec})
((lines_per_file = (total_lines + num_files - 1) / num_files))

# Split the actual file, maintaining lines.

split --lines=${lines_per_file} ${fspec} data.

# Debug information

echo -e "Split on ${num_files} files"
echo -e "Total lines = ${total_lines}"
echo -e "Lines per file = ${lines_per_file}"    
wc -l data.*
mkdir -p ./split
mv data.* ./split
