#!/usr/bin/env python3
"""
Export a TensorRT-compatible ONNX model
"""

import torch
import sys
import os

def export_tensorrt_compatible_model():
    """Export a model that's more compatible with TensorRT"""
    try:
        # Import the model
        from rfdetr.models.lwdetr import LWDETR
        from rfdetr.config import get_config
        
        print("Loading RF-DETR model...")
        
        # Get config
        config = get_config('rf-detr-base')
        
        # Create model
        model = LWDETR(config)
        
        # Load weights
        checkpoint = torch.load('rf-detr-base.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        print("Model loaded successfully")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Export with TensorRT-compatible settings
        print("Exporting ONNX model...")
        
        torch.onnx.export(
            model,
            dummy_input,
            "rf-detr-base-tensorrt-compatible.onnx",
            export_params=True,
            opset_version=11,  # Use older opset for better compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output_0', 'output_1'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output_0': {0: 'batch_size'},
                'output_1': {0: 'batch_size'}
            },
            verbose=True
        )
        
        print("✅ ONNX model exported successfully!")
        print("File: rf-detr-base-tensorrt-compatible.onnx")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        return False

if __name__ == "__main__":
    print("=== Exporting TensorRT-Compatible ONNX Model ===")
    success = export_tensorrt_compatible_model()
    
    if success:
        print("\nNext steps:")
        print("1. Try the new model with DeepStream")
        print("2. Or convert it using trtexec")
    else:
        print("\nExport failed. Check the error above.") 