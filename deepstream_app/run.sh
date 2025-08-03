#!/bin/bash

#
# RF-DETR DeepStream Application Runner
# This script helps build and run the DeepStream application
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RF-DETR DeepStream Application${NC}"
echo "=================================="

# Check if we're on a compatible system
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA not found. This application requires NVIDIA hardware.${NC}"
    exit 1
fi

# Check if DeepStream is installed
if [ ! -d "/opt/nvidia/deepstream" ]; then
    echo -e "${RED}Error: DeepStream not found. Please install NVIDIA DeepStream SDK.${NC}"
    exit 1
fi

# Check if ONNX model exists
if [ ! -f "../rf-detr-base.onnx" ]; then
    echo -e "${RED}Error: RF-DETR ONNX model not found at ../rf-detr-base.onnx${NC}"
    echo "Please ensure the model file is in the parent directory."
    exit 1
fi

# Check if camera is available
if [ ! -e "/dev/video0" ]; then
    echo -e "${YELLOW}Warning: Camera /dev/video0 not found.${NC}"
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "No video devices found"
    echo ""
fi

# Function to build the application
build_app() {
    echo -e "${GREEN}Building RF-DETR DeepStream Application...${NC}"
    make clean
    make all
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
    else
        echo -e "${RED}✗ Build failed!${NC}"
        exit 1
    fi
}

# Function to install the custom library
install_lib() {
    echo -e "${GREEN}Installing custom parsing library...${NC}"
    make install
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Library installed successfully!${NC}"
    else
        echo -e "${RED}✗ Library installation failed!${NC}"
        exit 1
    fi
}

# Function to run the application
run_app() {
    echo -e "${GREEN}Starting RF-DETR DeepStream Application...${NC}"
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    # Enable maximum performance on Jetson
    if command -v nvpmodel &> /dev/null; then
        echo "Enabling maximum performance mode..."
        sudo nvpmodel -m 0 2>/dev/null || true
        sudo jetson_clocks 2>/dev/null || true
    fi
    
    # Run the application
    ./rfdetr_deepstream_app
}

# Parse command line arguments
case "$1" in
    "build")
        build_app
        ;;
    "install")
        install_lib
        ;;
    "run")
        run_app
        ;;
    "setup")
        build_app
        install_lib
        echo -e "${GREEN}Setup complete! You can now run the application with:${NC}"
        echo "./run.sh run"
        ;;
    "clean")
        echo -e "${GREEN}Cleaning build artifacts...${NC}"
        make clean
        echo -e "${GREEN}✓ Clean complete!${NC}"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build    - Build the application and custom library"
        echo "  install  - Install the custom library system-wide"
        echo "  run      - Run the application"
        echo "  setup    - Build and install (complete setup)"
        echo "  clean    - Clean build artifacts"
        echo "  help     - Show this help message"
        echo ""
        echo "Default (no arguments): setup + run"
        ;;
    "")
        # Default action: setup and run
        build_app
        install_lib
        echo ""
        run_app
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac