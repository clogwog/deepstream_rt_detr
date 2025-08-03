#!/usr/bin/env python3
"""
Check TensorRT version and capabilities
"""

import subprocess
import sys

def check_tensorrt():
    """Check TensorRT installation and version"""
    print("=== TensorRT Version Check ===")
    
    # Check TensorRT version
    try:
        result = subprocess.run(['dpkg', '-l', '|', 'grep', 'tensorrt'], 
                              shell=True, capture_output=True, text=True)
        print("TensorRT packages:")
        print(result.stdout)
    except Exception as e:
        print(f"Error checking TensorRT packages: {e}")
    
    # Check if TensorRT Python is available
    try:
        import tensorrt as trt
        print(f"\nTensorRT Python version: {trt.__version__}")
        
        # Check available precision modes
        print(f"\nAvailable precision modes:")
        print(f"  FP32: {trt.float32}")
        print(f"  FP16: {trt.float16}")
        print(f"  INT8: {trt.int8}")
        
        # Check builder capabilities
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print(f"\nBuilder capabilities:")
        print(f"  FP16 available: {builder.platform_has_fast_fp16}")
        print(f"  INT8 available: {builder.platform_has_fast_int8}")
        
    except ImportError:
        print("TensorRT Python package not available")
    except Exception as e:
        print(f"Error checking TensorRT: {e}")

def check_deepstream():
    """Check DeepStream installation"""
    print("\n=== DeepStream Check ===")
    
    try:
        result = subprocess.run(['dpkg', '-l', '|', 'grep', 'deepstream'], 
                              shell=True, capture_output=True, text=True)
        print("DeepStream packages:")
        print(result.stdout)
    except Exception as e:
        print(f"Error checking DeepStream: {e}")

if __name__ == "__main__":
    check_tensorrt()
    check_deepstream() 