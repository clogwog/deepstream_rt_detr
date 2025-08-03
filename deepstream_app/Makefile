#
# Makefile for RF-DETR DeepStream Application
#

CXX = g++
NVCC = nvcc

APP_NAME = rfdetr_deepstream_app
LIB_NAME = libnvdsinfer_custom_impl_rfdetr.so

# Include directories
CFLAGS = -fPIC -Wall -Werror

# DeepStream and CUDA paths (adjust for your Jetson NX2 installation)
DS_INC = /opt/nvidia/deepstream/deepstream/sources/includes
CUDA_INC = /usr/local/cuda/include
GSTREAMER_INC = /usr/include/gstreamer-1.0
GLIB_INC = /usr/include/glib-2.0
GLIB_CONFIG_INC = /usr/lib/aarch64-linux-gnu/glib-2.0/include

INCLUDES = -I$(DS_INC) \
           -I$(CUDA_INC) \
           -I$(GSTREAMER_INC) \
           -I$(GLIB_INC) \
           -I$(GLIB_CONFIG_INC)

# Library paths
DS_LIB = /opt/nvidia/deepstream/deepstream/lib
CUDA_LIB = /usr/local/cuda/lib64

LIBS = -L$(DS_LIB) -L$(CUDA_LIB) \
       -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord -lnvds_utils \
       -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 \
       -lcuda -lcudart

CXXFLAGS = $(CFLAGS) $(INCLUDES) -std=c++14

# Source files
APP_SRCS = main.cpp
LIB_SRCS = nvdsinfer_custom_impl_rfdetr.cpp

# Object files
APP_OBJS = $(APP_SRCS:.cpp=.o)
LIB_OBJS = $(LIB_SRCS:.cpp=.o)

.PHONY: all clean

all: $(APP_NAME) $(LIB_NAME)

# Build main application
$(APP_NAME): $(APP_OBJS)
	$(CXX) -o $@ $^ $(LIBS)

# Build custom parsing library
$(LIB_NAME): $(LIB_OBJS)
	$(CXX) -shared -o $@ $^ $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Clean build artifacts
clean:
	rm -f $(APP_OBJS) $(LIB_OBJS) $(APP_NAME) $(LIB_NAME)

# Install (copy files to appropriate locations)
install: all
	sudo cp $(LIB_NAME) /usr/lib/
	sudo ldconfig
	@echo "Custom library installed to /usr/lib/"
	@echo "You can now run: ./$(APP_NAME)"

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: all

# Help
help:
	@echo "RF-DETR DeepStream Application Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build both application and custom library"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install custom library system-wide"
	@echo "  debug      - Build with debug symbols"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Before building, make sure you have:"
	@echo "  - NVIDIA DeepStream SDK installed"
	@echo "  - CUDA toolkit installed"
	@echo "  - GStreamer development packages"
	@echo "  - RF-DETR ONNX model (rf-detr-base.onnx) in parent directory"