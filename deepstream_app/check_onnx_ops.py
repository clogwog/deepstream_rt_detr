#!/usr/bin/env python3
"""
Check ONNX model operations to understand what needs to be converted
"""

import onnx
import sys

def check_onnx_operations(model_path):
    """Check what operations are in the ONNX model"""
    try:
        model = onnx.load(model_path)
        print(f"ONNX Model: {model_path}")
        print(f"IR Version: {model.ir_version}")
        print(f"Opset Version: {model.opset_import[0].version}")
        print(f"Producer: {model.producer_name}")
        print()
        
        # Get all unique operation types
        op_types = set()
        for node in model.graph.node:
            op_types.add(node.op_type)
        
        print("Operations found in model:")
        for op_type in sorted(op_types):
            count = sum(1 for node in model.graph.node if node.op_type == op_type)
            print(f"  {op_type}: {count} instances")
        
        # Check for problematic operations
        problematic_ops = ['LayerNormalization', 'Gelu', 'Softmax']
        print("\nProblematic operations for TensorRT:")
        for op in problematic_ops:
            if op in op_types:
                print(f"  ❌ {op} - Not supported by TensorRT")
            else:
                print(f"  ✅ {op} - Not found")
                
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    model_path = "rf-detr-base.onnx"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    check_onnx_operations(model_path) 