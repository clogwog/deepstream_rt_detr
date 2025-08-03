#!/usr/bin/env python3
"""
Debug script to inspect RF-DETR ONNX model structure
"""

import onnx
import numpy as np

def inspect_onnx_model(model_path):
    """Inspect ONNX model structure"""
    print(f"Loading ONNX model: {model_path}")
    
    try:
        # Load the model
        model = onnx.load(model_path)
        
        print(f"\n=== ONNX Model Info ===")
        print(f"Model IR version: {model.ir_version}")
        print(f"Opset version: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name}")
        
        print(f"\n=== Inputs ===")
        for i, input_info in enumerate(model.graph.input):
            print(f"Input {i}: {input_info.name}")
            print(f"  Shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}")
            print(f"  Type: {input_info.type.tensor_type.elem_type}")
        
        print(f"\n=== Outputs ===")
        for i, output_info in enumerate(model.graph.output):
            print(f"Output {i}: {output_info.name}")
            print(f"  Shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}")
            print(f"  Type: {output_info.type.tensor_type.elem_type}")
        
        print(f"\n=== Model Size ===")
        import os
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model file size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

if __name__ == "__main__":
    # Check if model exists
    import os
    model_path = "../rf-detr-base.onnx"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Available files:")
        for f in os.listdir("."):
            if f.endswith(".onnx"):
                print(f"  {f}")
    else:
        inspect_onnx_model(model_path) 