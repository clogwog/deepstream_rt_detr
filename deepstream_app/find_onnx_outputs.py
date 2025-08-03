#!/usr/bin/env python3
"""
Find ONNX model output names using basic file parsing
"""

import os
import re

def find_onnx_outputs():
    """Find output names in ONNX model file"""
    print("Searching for ONNX model output names...")
    print("=" * 50)
    
    model_path = "../rf-detr-base.onnx"
    
    if not os.path.exists(model_path):
        print("Model not found at: {}".format(model_path))
        return False
    
    try:
        with open(model_path, 'rb') as f:
            content = f.read()
            
        # Convert to string for pattern matching
        content_str = content.decode('utf-8', errors='ignore')
        
        print("Searching for output patterns...")
        
        # Common output name patterns
        output_patterns = [
            r'pred_boxes',
            r'pred_logits', 
            r'output',
            r'dets',
            r'labels',
            r'boxes',
            r'scores',
            r'logits',
            r'predictions',
            r'output_0',
            r'output_1',
            r'output_2',
            r'output_3'
        ]
        
        found_outputs = []
        
        for pattern in output_patterns:
            matches = re.findall(pattern, content_str, re.IGNORECASE)
            if matches:
                found_outputs.extend(matches)
                print("Found pattern '{}': {} times".format(pattern, len(matches)))
        
        # Remove duplicates and show unique outputs
        unique_outputs = list(set(found_outputs))
        print("\nUnique output names found:")
        for i, output in enumerate(unique_outputs):
            print("  {}. {}".format(i+1, output))
        
        # Look for input names too
        print("\nSearching for input patterns...")
        input_patterns = [
            r'input',
            r'image',
            r'data',
            r'x',
            r'input_0',
            r'input_1'
        ]
        
        found_inputs = []
        for pattern in input_patterns:
            matches = re.findall(pattern, content_str, re.IGNORECASE)
            if matches:
                found_inputs.extend(matches)
                print("Found input pattern '{}': {} times".format(pattern, len(matches)))
        
        unique_inputs = list(set(found_inputs))
        print("\nUnique input names found:")
        for i, input_name in enumerate(unique_inputs):
            print("  {}. {}".format(i+1, input_name))
        
        # Suggest config
        if unique_outputs:
            print("\nSuggested config output-blob-names:")
            if len(unique_outputs) >= 2:
                print("output-blob-names={};{}".format(unique_outputs[0], unique_outputs[1]))
            else:
                print("output-blob-names={}".format(unique_outputs[0]))
        
        return True
        
    except Exception as e:
        print("Error reading model: {}".format(e))
        return False

if __name__ == "__main__":
    find_onnx_outputs() 