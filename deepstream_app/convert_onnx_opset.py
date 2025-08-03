#!/usr/bin/env python3
"""
Convert ONNX model to older opset version for better TensorRT compatibility
"""

import subprocess
import sys
import os

def convert_onnx_opset():
    """Convert ONNX model to opset 11 for better TensorRT compatibility"""
    print("Converting ONNX model to opset 11...")
    print("=" * 50)
    
    # Check if onnx-simplifier is available
    try:
        import onnx
        print("OK: ONNX package available")
    except ImportError:
        print("Warning: ONNX package not available, trying pip install...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "onnx"], check=True)
            import onnx
            print("OK: ONNX package installed")
        except Exception as e:
            print("Failed to install ONNX: {}".format(e))
            return False
    
    # Load and convert the model
    try:
        print("Loading ONNX model...")
        model = onnx.load("../rf-detr-base.onnx")
        
        print("Current opset version: {}".format(model.opset_import[0].version))
        
        # Convert to opset 11
        print("Converting to opset 11...")
        from onnx import version_converter
        
        try:
            converted_model = version_converter.convert_version(model, 11)
            print("OK: Successfully converted to opset 11")
        except Exception as e:
            print("Warning: Version conversion failed: {}".format(e))
            print("Trying opset 12...")
            try:
                converted_model = version_converter.convert_version(model, 12)
                print("OK: Successfully converted to opset 12")
            except Exception as e2:
                print("Failed: Both opset 11 and 12 conversion failed")
                print("Error 11: {}".format(e))
                print("Error 12: {}".format(e2))
                return False
        
        # Save the converted model
        output_path = "../rf-detr-base-opset11.onnx"
        onnx.save(converted_model, output_path)
        print("OK: Saved converted model to: {}".format(output_path))
        
        # Check file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print("Converted model size: {:.2f} MB".format(size_mb))
        
        return True
        
    except Exception as e:
        print("Error during conversion: {}".format(e))
        return False

if __name__ == "__main__":
    convert_onnx_opset() 