#!/usr/bin/env python3
"""
Test ONNX Runtime availability and model loading
"""

import sys
import os

def test_onnxruntime():
    """Test if ONNX Runtime is available"""
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime is available")
        print(f"Version: {ort.__version__}")
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        return True
    except ImportError:
        print("❌ ONNX Runtime is not available")
        print("Try installing with: pip install onnxruntime")
        return False

def test_model_loading(model_path):
    """Test if we can load the ONNX model with ONNX Runtime"""
    try:
        import onnxruntime as ort
        
        print(f"\nTesting model loading: {model_path}")
        
        # Create inference session
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully with ONNX Runtime")
        
        # Get input/output info
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")
        
        # Get input shape
        input_shape = session.get_inputs()[0].shape
        print(f"Input shape: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_tensorrt_availability():
    """Test if TensorRT is available"""
    try:
        import tensorrt as trt
        print("✅ TensorRT is available")
        print(f"Version: {trt.__version__}")
        return True
    except ImportError:
        print("❌ TensorRT is not available")
        return False

if __name__ == "__main__":
    model_path = "rf-detr-base.onnx"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("=== Environment Check ===")
    
    # Check ONNX Runtime
    onnxruntime_available = test_onnxruntime()
    
    # Check TensorRT
    tensorrt_available = test_tensorrt_availability()
    
    # Test model loading if ONNX Runtime is available
    if onnxruntime_available:
        test_model_loading(model_path)
    
    print("\n=== Recommendations ===")
    if onnxruntime_available:
        print("✅ Use ONNX Runtime - it should handle LayerNormalization")
    else:
        print("❌ Install ONNX Runtime: pip install onnxruntime")
    
    if not tensorrt_available:
        print("❌ TensorRT not available - this explains the LayerNormalization errors") 