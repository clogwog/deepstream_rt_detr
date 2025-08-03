#!/bin/bash

# Convert ONNX to TensorRT Engine
# This script converts the RF-DETR ONNX model to a TensorRT engine

echo "Converting RF-DETR ONNX to TensorRT Engine..."
echo "=============================================="

# Check if ONNX file exists
if [ ! -f "../rf-detr-base.onnx" ]; then
    echo "Error: rf-detr-base.onnx not found in parent directory"
    exit 1
fi

# Check if trtexec exists
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    echo "Error: trtexec not found at $TRTEXEC"
    exit 1
fi

echo "Found trtexec at: $TRTEXEC"
echo "ONNX file: ../rf-detr-base.onnx"
echo ""

# Try FP16 first (faster, less memory)
echo "Attempting FP16 conversion..."
$TRTEXEC --onnx=../rf-detr-base.onnx --saveEngine=../rf-detr-base.engine --fp16

if [ $? -eq 0 ]; then
    echo "✓ FP16 conversion successful!"
    echo "TensorRT engine saved as: ../rf-detr-base.engine"
else
    echo "⚠ FP16 conversion failed, trying FP32..."
    
    # Try FP32 if FP16 fails
    $TRTEXEC --onnx=../rf-detr-base.onnx --saveEngine=../rf-detr-base.engine --fp32
    
    if [ $? -eq 0 ]; then
        echo "✓ FP32 conversion successful!"
        echo "TensorRT engine saved as: ../rf-detr-base.engine"
    else
        echo "✗ Both FP16 and FP32 conversions failed"
        echo "Check the error messages above for details"
        exit 1
    fi
fi

echo ""
echo "Next steps:"
echo "1. Copy the TensorRT config: cp config_infer_primary_tensorrt.txt config_infer_primary.txt"
echo "2. Run the DeepStream app: ./rfdetr_deepstream_app" 