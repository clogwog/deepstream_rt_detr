# RF-DETR DeepStream Implementation

This repository contains a high-performance DeepStream implementation of RF-DETR object detection for NVIDIA Jetson devices.

## 🚀 Quick Start

The main application is in the `deepstream_app/` directory:

```bash
cd deepstream_app
./run.sh  # Build and run
```

## 📁 Contents

- **`deepstream_app/`** - Complete C++ DeepStream application
  - Real-time camera processing with hardware acceleration
  - RF-DETR ONNX model inference via TensorRT
  - COCO-91 to COCO-80 class mapping
  - Optimized for NVIDIA Jetson NX2

## 🎯 Features

- ⚡ **60-120 FPS** performance on Jetson NX2
- 🎥 **Live camera processing** (`/dev/video0`)
- 🎯 **Accurate object detection** with proper class mapping
- 📊 **Real-time visualization** with bounding boxes
- 🔧 **Easy setup** with automated build scripts

## 📋 Requirements

- NVIDIA Jetson NX2 (or compatible GPU device)
- NVIDIA DeepStream SDK 6.x
- RF-DETR ONNX model
- Camera connected to `/dev/video0`

See `deepstream_app/requirements.md` for detailed setup instructions.

## 🔧 Performance

Expected performance on Jetson NX2:
- **Inference**: ~8-15ms per frame
- **Total pipeline**: ~8-16ms per frame  
- **Frame rate**: 60-120 FPS

## 📚 Documentation

- `deepstream_app/README.md` - Quick start guide
- `deepstream_app/requirements.md` - Detailed setup and troubleshooting
- `deepstream_app/Makefile` - Build system documentation

Built with NVIDIA DeepStream SDK for maximum performance! 🚀