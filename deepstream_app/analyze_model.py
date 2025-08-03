#!/usr/bin/env python3
"""
Simple ONNX model analyzer that doesn't require the onnx package
"""

import subprocess
import sys
import os

def check_netron():
    """Check if netron is available"""
    try:
        result = subprocess.run(['netron', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_netron():
    """Install netron if not available"""
    try:
        print("Installing netron...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'netron'], 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install netron")
        return False

def analyze_model_with_netron(model_path):
    """Analyze ONNX model using netron"""
    if not check_netron():
        if not install_netron():
            print("Cannot install netron. Please install manually:")
            print("pip install netron")
            return False
    
    try:
        print(f"Analyzing model: {model_path}")
        print("Opening netron viewer...")
        subprocess.run(['netron', model_path])
        return True
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return False

def check_model_basic(model_path):
    """Basic model check without external dependencies"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"Model file: {model_path}")
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Try to read the first few bytes to check if it's a valid file
    try:
        with open(model_path, 'rb') as f:
            header = f.read(8)
            if header.startswith(b'ONNX'):
                print("✅ Valid ONNX model file detected")
                return True
            else:
                print("❌ Not a valid ONNX model file")
                return False
    except Exception as e:
        print(f"Error reading model file: {e}")
        return False

if __name__ == "__main__":
    model_path = "rf-detr-base.onnx"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("=== ONNX Model Analysis ===")
    
    # Basic check
    if check_model_basic(model_path):
        print("\n=== Detailed Analysis ===")
        analyze_model_with_netron(model_path)
    else:
        print("Model file check failed") 