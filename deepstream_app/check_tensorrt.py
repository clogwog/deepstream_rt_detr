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
        print("\nTensorRT Python version: {}".format(trt.__version__))
        
        # Check available precision modes
        print("\nAvailable precision modes:")
        print("  FP32: {}".format(trt.float32))
        print("  FP16: {}".format(trt.float16))
        print("  INT8: {}".format(trt.int8))
        
        # Check builder capabilities
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print("\nBuilder capabilities:")
        print("  FP16 available: {}".format(builder.platform_has_fast_fp16))
        print("  INT8 available: {}".format(builder.platform_has_fast_int8))
        
    except ImportError:
        print("TensorRT Python package not available")
    except Exception as e:
        print("Error checking TensorRT: {}".format(e))

def check_deepstream():
    """Check DeepStream installation"""
    print("\n=== DeepStream Check ===")
    
    try:
        result = subprocess.run(['dpkg', '-l', '|', 'grep', 'deepstream'], 
                              shell=True, capture_output=True, text=True)
        print("DeepStream packages:")
        print(result.stdout)
    except Exception as e:
        print("Error checking DeepStream: {}".format(e))

if __name__ == "__main__":
    check_tensorrt()
    check_deepstream() 