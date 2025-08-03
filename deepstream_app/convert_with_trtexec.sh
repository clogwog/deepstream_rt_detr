#!/bin/bash
#
# Convert ONNX model to TensorRT engine using trtexec
# This should handle LayerNormalization operations better
#

set -e

MODEL_PATH="rf-detr-base.onnx"
ENGINE_PATH="rf-detr-base-tensorrt.engine"
WORKSPACE_SIZE="4096"  # 4GB workspace

echo "=== Converting ONNX model to TensorRT engine ==="
echo "Input model: $MODEL_PATH"
echo "Output engine: $ENGINE_PATH"
echo "Workspace size: ${WORKSPACE_SIZE}MB"

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "❌ trtexec not found. Please install TensorRT or add it to PATH"
    echo "Try: export PATH=/usr/local/tensorrt/bin:$PATH"
    exit 1
fi

# Check if input model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Input model not found: $MODEL_PATH"
    exit 1
fi

echo "✅ Starting conversion..."

# Convert ONNX to TensorRT engine
trtexec \
    --onnx="$MODEL_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --workspace=$WORKSPACE_SIZE \
    --fp16 \
    --verbose \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:1x3x640x640 \
    --maxShapes=input:1x3x640x640

if [ $? -eq 0 ]; then
    echo "✅ Conversion successful!"
    echo "Engine saved to: $ENGINE_PATH"
    
    # Show engine info
    echo "=== Engine Information ==="
    trtexec --loadEngine="$ENGINE_PATH" --dumpProfile
else
    echo "❌ Conversion failed!"
    exit 1
fi 