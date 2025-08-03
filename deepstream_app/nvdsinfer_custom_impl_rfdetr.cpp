/*
 * Custom implementation for RF-DETR ONNX model parsing
 * This file implements the custom bounding box parsing function for RF-DETR
 * that handles COCO-91 to COCO-80 class mapping and coordinate conversion
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* COCO-91 to COCO-80 mapping (same as Python version) */
static const std::unordered_map<int, int> coco_91_to_80_map = {
    {1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}, {6, 5}, {7, 6}, {8, 7}, {9, 8}, {10, 9},
    {11, 10}, {13, 11}, {14, 12}, {15, 13}, {16, 14}, {17, 15}, {18, 16}, {19, 17}, {20, 18},
    {21, 19}, {22, 20}, {23, 21}, {24, 22}, {25, 23}, {27, 24}, {28, 25}, {31, 26}, {32, 27},
    {33, 28}, {34, 29}, {35, 30}, {36, 31}, {37, 32}, {38, 33}, {39, 34}, {40, 35}, {41, 36},
    {42, 37}, {43, 38}, {44, 39}, {46, 40}, {47, 41}, {48, 42}, {49, 43}, {50, 44}, {51, 45},
    {52, 46}, {53, 47}, {54, 48}, {55, 49}, {56, 50}, {57, 51}, {58, 52}, {59, 53}, {60, 54},
    {61, 55}, {62, 56}, {63, 57}, {64, 58}, {65, 59}, {67, 60}, {70, 61}, {72, 62}, {73, 63},
    {74, 64}, {75, 65}, {76, 66}, {77, 67}, {78, 68}, {79, 69}, {80, 70}, {81, 71}, {82, 72},
    {84, 73}, {85, 74}, {86, 75}, {87, 76}, {88, 77}, {89, 78}, {90, 79}
};

/* Sigmoid function */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-CLIP(x, -250.0f, 250.0f)));
}

/* Convert bounding box from (center_x, center_y, width, height) to (x1, y1, x2, y2) */
static void convertBBoxCxCyWhToXyXy(float cx, float cy, float w, float h, 
                                    float& x1, float& y1, float& x2, float& y2) {
    x1 = cx - 0.5f * w;
    y1 = cy - 0.5f * h;
    x2 = cx + 0.5f * w;
    y2 = cy + 0.5f * h;
}

/* Structure to hold detection data */
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

/* Custom RF-DETR parsing function */
extern "C" bool NvDsInferParseCustomRFDETR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.size() != 2) {
        std::cerr << "Expected 2 output layers (dets, labels), got " << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Get output layer information */
    const NvDsInferLayerInfo& detsLayer = outputLayersInfo[0];  // "dets"
    const NvDsInferLayerInfo& labelsLayer = outputLayersInfo[1]; // "labels"
    
    /* Verify dimensions */
    if (detsLayer.inferDims.numDims != 3 || labelsLayer.inferDims.numDims != 3) {
        std::cerr << "Invalid output dimensions" << std::endl;
        return false;
    }
    
    int batch_size = detsLayer.inferDims.d[0];
    int num_queries = detsLayer.inferDims.d[1];
    int bbox_dims = detsLayer.inferDims.d[2];  // Should be 4 (cx, cy, w, h)
    int num_classes = labelsLayer.inferDims.d[2];  // Should be 91 for COCO-91
    
    if (bbox_dims != 4) {
        std::cerr << "Expected 4 bbox dimensions (cx, cy, w, h), got " << bbox_dims << std::endl;
        return false;
    }
    
    /* Get raw output data */
    const float* dets_data = static_cast<const float*>(detsLayer.buffer);
    const float* labels_data = static_cast<const float*>(labelsLayer.buffer);
    
    /* Process for first batch only (batch_size should be 1) */
    std::vector<Detection> detections;
    
    /* Calculate confidence scores using sigmoid */
    std::vector<float> confidences(num_queries * num_classes);
    for (int i = 0; i < num_queries * num_classes; i++) {
        confidences[i] = sigmoid(labels_data[i]);
    }
    
    /* Find top detections */
    std::vector<std::pair<float, int>> score_index_pairs;
    for (int i = 0; i < num_queries * num_classes; i++) {
        if (confidences[i] >= detectionParams.perClassPreclusterThreshold[0]) {
            score_index_pairs.push_back(std::make_pair(confidences[i], i));
        }
    }
    
    /* Sort by confidence score (descending) */
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    /* Limit to maximum detections */
    int max_detections = std::min((int)score_index_pairs.size(), 
                                  (int)detectionParams.perClassPreclusterThreshold.size() * 100);
    
    /* Process detections */
    for (int i = 0; i < max_detections; i++) {
        float confidence = score_index_pairs[i].first;
        int flat_index = score_index_pairs[i].second;
        
        int query_idx = flat_index / num_classes;
        int coco_91_class = flat_index % num_classes;
        
        /* Skip if confidence is too low */
        if (confidence < detectionParams.perClassPreclusterThreshold[0]) {
            continue;
        }
        
        /* Map COCO-91 to COCO-80 */
        auto it = coco_91_to_80_map.find(coco_91_class);
        if (it == coco_91_to_80_map.end()) {
            continue; // Skip if not in mapping (background or unused class)
        }
        int coco_80_class = it->second;
        
        /* Get bounding box coordinates */
        int bbox_offset = query_idx * 4;
        float cx = dets_data[bbox_offset + 0];
        float cy = dets_data[bbox_offset + 1];
        float w = dets_data[bbox_offset + 2];
        float h = dets_data[bbox_offset + 3];
        
        /* Convert to (x1, y1, x2, y2) format */
        float x1, y1, x2, y2;
        convertBBoxCxCyWhToXyXy(cx, cy, w, h, x1, y1, x2, y2);
        
        /* Scale to image coordinates */
        float img_width = networkInfo.width;
        float img_height = networkInfo.height;
        
        x1 *= img_width;
        y1 *= img_height;
        x2 *= img_width;
        y2 *= img_height;
        
        /* Clip to image bounds */
        x1 = CLIP(x1, 0.0f, img_width);
        y1 = CLIP(y1, 0.0f, img_height);
        x2 = CLIP(x2, 0.0f, img_width);
        y2 = CLIP(y2, 0.0f, img_height);
        
        /* Create detection object */
        NvDsInferParseObjectInfo obj;
        obj.classId = coco_80_class;
        obj.detectionConfidence = confidence;
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        
        objectList.push_back(obj);
    }
    
    return true;
}

/* Check if custom function is correctly called */
extern "C" bool NvDsInferParseCustomRFDETRTest() {
    std::cout << "RF-DETR custom parsing function is loaded correctly!" << std::endl;
    return true;
}