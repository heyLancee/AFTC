#!/bin/bash

# ./convert.sh ./models/actor.pth ./models/actor.onnx 10 1 cpu "400,300" 3

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 获取项目根目录（脚本目录的上一级）
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# 检查参数数量
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <pth_model_path> <onnx_output_path> <state_dim> [batch_size] [device] [hidden_size] [output_dim]"
    echo "Example: $0 ./models/actor.pth ./models/actor.onnx 10 1 cpu \"400,300\" 3"
    exit 1
fi

# 获取参数
PTH_PATH="$1"
ONNX_PATH="$2"
STATE_DIM="$3"
BATCH_SIZE="${4:-1}"       # 默认batch_size为1
DEVICE="${5:-cpu}"         # 默认device为cpu
HIDDEN_SIZE="${6:-256,256}" # 默认hidden_size为"256,256"
OUTPUT_DIM="${7:-3}"       # 默认output_dim为3

# 检查.pth文件是否存在
if [ ! -f "$PTH_PATH" ]; then
    echo "Error: PyTorch model file '$PTH_PATH' not found!"
    exit 1
fi

# 创建输出目录（如果不存在）
OUTPUT_DIR=$(dirname "$ONNX_PATH")
mkdir -p "$OUTPUT_DIR"

# 执行转换（使用完整路径）
python "$PROJECT_ROOT/pth2ONNX.py" \
    --pth_path "$PTH_PATH" \
    --onnx_path "$ONNX_PATH" \
    --state_dim "$STATE_DIM" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hidden_size "$HIDDEN_SIZE" \
    --output_dim "$OUTPUT_DIM"

# 检查转换是否成功
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    echo "ONNX model saved to: $ONNX_PATH"
else
    echo "Error: Conversion failed!"
    exit 1
fi
