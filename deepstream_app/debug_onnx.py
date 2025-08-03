#!/usr/bin/env python3
"""
Debug script to inspect RF-DETR ONNX model structure
"""

import onnx
import numpy as np

def inspect_onnx_model(model_path):
    """Inspect ONNX model structure"""
    print("Loading ONNX model: {}".format(model_path))
    
    try:
        # Load the model
        model = onnx.load(model_path)
        
        print("\n=== ONNX Model Info ===")
        print("Model IR version: {}".format(model.ir_version))
        print("Opset version: {}".format(model.opset_import[0].version))
        print("Producer: {}".format(model.producer_name))
        
        print("\n=== Inputs ===")
        for i, input_info in enumerate(model.graph.input):
            print("Input {}: {}".format(i, input_info.name))
            print("  Shape: {}".format([dim.dim_value for dim in input_info.type.tensor_type.shape.dim]))
            print("  Type: {}".format(input_info.type.tensor_type.elem_type))
        
        print("\n=== Outputs ===")
        for i, output_info in enumerate(model.graph.output):
            print("Output {}: {}".format(i, output_info.name))
            print("  Shape: {}".format([dim.dim_value for dim in output_info.type.tensor_type.shape.dim]))
            print("  Type: {}".format(output_info.type.tensor_type.elem_type))
        
        print("\n=== Model Size ===")
        import os
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print("Model file size: {:.2f} MB".format(size_mb))
        
        return True
        
    except Exception as e:
        print("Error loading ONNX model: {}".format(e))
        return False

if __name__ == "__main__":
    # Check if model exists
    import os
    model_path = "../rf-detr-base.onnx"
    
    if not os.path.exists(model_path):
        print("Model not found at: {}".format(model_path))
        print("Available files:")
        for f in os.listdir("."):
            if f.endswith(".onnx"):
                print("  {}".format(f))
    else:
        inspect_onnx_model(model_path) 