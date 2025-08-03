# RF-DETR DeepStream Application

This is a high-performance DeepStream implementation of the RF-DETR object detection pipeline, designed to run on NVIDIA Jetson devices with hardware acceleration.

## Quick Start

```bash
# Build and run (first time setup)
./run.sh

# Or step by step:
./run.sh setup    # Build and install
./run.sh run      # Run the application
```

## Features

- 🚀 **Real-time camera processing** with hardware acceleration
- 🎯 **RF-DETR ONNX model** inference via TensorRT
- 📊 **COCO-91 to COCO-80 class mapping** (same as Python version)
- 🎥 **Live visualization** with bounding boxes and confidence scores
- ⚡ **Optimized performance** (60-120 FPS on Jetson NX2)

## Equivalent Functionality

This DeepStream application replicates:
```bash
python ultimate_rfdetr_pipeline.py --camera 0
```

## Files

- `main.cpp` - Main DeepStream application
- `nvdsinfer_custom_impl_rfdetr.cpp` - Custom RF-DETR parsing logic
- `config_infer_primary.txt` - nvinfer configuration for ONNX model
- `labels.txt` - COCO-80 class names
- `Makefile` - Build system
- `run.sh` - Helper script for building and running
- `requirements.md` - Detailed setup and troubleshooting guide

## Requirements

- NVIDIA Jetson NX2 (or compatible GPU device)
- NVIDIA DeepStream SDK 6.x
- RF-DETR ONNX model (`../rf-detr-base.onnx`)
- Camera connected to `/dev/video0`

## Performance

Expected performance on Jetson NX2:
- **Inference**: ~8-15ms per frame
- **Total pipeline**: ~8-16ms per frame
- **Frame rate**: 60-120 FPS (depending on input resolution)

For detailed setup instructions, see `requirements.md`.