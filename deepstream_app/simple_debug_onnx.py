#!/usr/bin/env python3
"""
Simple ONNX model inspection without requiring the onnx package
"""

import os
import struct
import sys

def inspect_onnx_file_simple(model_path):
    """Simple inspection of ONNX file without the onnx package"""
    print("Simple ONNX Model Inspection")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print("Model not found at: {}".format(model_path))
        print("Available files:")
        for f in os.listdir("."):
            if f.endswith(".onnx"):
                print("  {}".format(f))
        return False
    
    # Get file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print("Model file size: {:.2f} MB".format(size_mb))
    
    # Try to read the file header
    try:
        with open(model_path, 'rb') as f:
            # Read first few bytes to check if it's a valid file
            header = f.read(100)
            
            # Look for common ONNX patterns
            if b'onnx' in header.lower():
                print("✓ File appears to be a valid ONNX model")
            else:
                print("⚠ File may not be a valid ONNX model")
            
            # Try to find input/output information in the binary
            f.seek(0)
            content = f.read()
            
            # Look for common input/output names
            input_patterns = [b'input', b'Input', b'INPUT']
            output_patterns = [b'output', b'Output', b'OUTPUT', b'dets', b'labels', b'boxes', b'scores']
            
            print("\nSearching for input/output patterns...")
            
            for pattern in input_patterns:
                if pattern in content:
                    print("Found input pattern: {}".format(pattern))
            
            for pattern in output_patterns:
                if pattern in content:
                    print("Found output pattern: {}".format(pattern))
            
            # Look for shape information
            shape_patterns = [b'640', b'3', b'batch', b'Batch']
            print("\nSearching for shape information...")
            for pattern in shape_patterns:
                if pattern in content:
                    print("Found shape pattern: {}".format(pattern))
                    
    except Exception as e:
        print("Error reading file: {}".format(e))
        return False
    
    return True

def check_tensorrt_simple():
    """Simple TensorRT check without the tensorrt package"""
    print("\n" + "=" * 40)
    print("Simple TensorRT Check")
    print("=" * 40)
    
    # Check if TensorRT is installed
    try:
        import subprocess
        result = subprocess.run(['dpkg', '-l', '|', 'grep', 'tensorrt'], 
                              shell=True, capture_output=True, text=True)
        print("TensorRT packages:")
        print(result.stdout)
    except Exception as e:
        print("Error checking TensorRT packages: {}".format(e))
    
    # Check DeepStream
    try:
        result = subprocess.run(['dpkg', '-l', '|', 'grep', 'deepstream'], 
                              shell=True, capture_output=True, text=True)
        print("\nDeepStream packages:")
        print(result.stdout)
    except Exception as e:
        print("Error checking DeepStream: {}".format(e))
    
    # Check Python version
    print("\nPython version: {}".format(sys.version))
    
    # Check if we can import any TensorRT related modules
    try:
        import ctypes
        # Try to load TensorRT library
        try:
            trt_lib = ctypes.CDLL("libnvinfer.so")
            print("✓ TensorRT library found: libnvinfer.so")
        except:
            print("⚠ TensorRT library not found: libnvinfer.so")
    except Exception as e:
        print("Error checking TensorRT library: {}".format(e))

if __name__ == "__main__":
    model_path = "rf-detr-base.onnx"
    
    print("Simple ONNX Model Debug Tool")
    print("=" * 50)
    
    # Inspect the model file
    inspect_onnx_file_simple(model_path)
    
    # Check TensorRT
    check_tensorrt_simple()
    
    print("\n" + "=" * 50)
    print("Next steps:")
    print("1. Try running the DeepStream app with auto-detection")
    print("2. Check the DeepStream logs for more specific error messages")
    print("3. Consider using a pre-built TensorRT engine instead of ONNX") 