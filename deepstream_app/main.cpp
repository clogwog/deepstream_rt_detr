/*
 * RF-DETR DeepStream Application
 * 
 * This application replicates the live camera functionality of the Python
 * ultimate_rfdetr_pipeline.py using NVIDIA DeepStream SDK.
 * 
 * Features:
 * - Live camera input (equivalent to --camera 0)
 * - ONNX model inference using nvinfer
 * - COCO-91 to COCO-80 class mapping
 * - Real-time object detection visualization
 * - Correct bounding box handling (cxcywh -> xyxy)
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_BICYCLE 1
#define PGIE_CLASS_ID_PERSON 2
#define PGIE_CLASS_ID_ROADSIGN 3

#define PGIE_DETECTED_CLASS_NUM 80

/* COCO-80 class names (same as Python version) */
static const gchar* coco_class_names[PGIE_DETECTED_CLASS_NUM] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

/* COCO-91 to COCO-80 mapping (same as Python version) */
static const gint coco_91_to_80_map[] = {
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, -1, 24, 25, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, -1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, -1, -1, 60, -1, -1, -1, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, -1, 73, 74, 75, 76, 77, 78, 79, -1
};
static const int COCO_91_TO_80_MAP_SIZE = sizeof(coco_91_to_80_map) / sizeof(coco_91_to_80_map[0]);

/* Structure to hold application data */
typedef struct {
    GstElement *pipeline;
    GstElement *source;
    GstElement *h264parser;
    GstElement *decoder;
    GstElement *streammux;
    GstElement *pgie;
    GstElement *nvvidconv;
    GstElement *nvosd;
    GstElement *transform;
    GstElement *sink;
    
    GMainLoop *loop;
    guint bus_watch_id;
} AppCtx;

/* Convert COCO-91 class index to COCO-80 class index */
static gint map_coco_91_to_80(gint coco_91_idx) {
    if (coco_91_idx >= 0 && coco_91_idx < COCO_91_TO_80_MAP_SIZE) {
        return coco_91_to_80_map[coco_91_idx];
    }
    return -1;
}

/* osd_sink_pad_buffer_probe: Callback function for processing detection metadata */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        
        /* Process each detected object */
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            
            /* Map COCO-91 class to COCO-80 class */
            gint coco_80_class = map_coco_91_to_80(obj_meta->class_id);
            
            if (coco_80_class >= 0 && coco_80_class < PGIE_DETECTED_CLASS_NUM) {
                /* Update class_id to COCO-80 index */
                obj_meta->class_id = coco_80_class;
                
                /* Set the correct class name */
                strcpy(obj_meta->obj_label, coco_class_names[coco_80_class]);
                
                /* Count specific objects for display */
                if (coco_80_class == 0) { /* person */
                    person_count++;
                } else if (coco_80_class == 2) { /* car */
                    vehicle_count++;
                }
                
                printf("Object: %s, Confidence: %.2f, BBox: (%.2f, %.2f, %.2f, %.2f)\n",
                       obj_meta->obj_label, obj_meta->confidence,
                       obj_meta->rect_params.left, obj_meta->rect_params.top,
                       obj_meta->rect_params.width, obj_meta->rect_params.height);
            }
        }
        
        /* Add display metadata for frame-level information */
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        
        txt_params->display_text = (gchar *) g_malloc0(MAX_DISPLAY_LEN);
        snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person: %d, Vehicle: %d", 
                 person_count, vehicle_count);
        
        /* Set text properties */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;
        txt_params->font_params.font_name = g_strdup("Serif");
        txt_params->font_params.font_size = 14;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;
        
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
    
    return GST_PAD_PROBE_OK;
}

/* Bus callback function */
static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    
    return TRUE;
}

/* Signal handler for Ctrl+C */
static void
sig_handler(int signum)
{
    g_print("\nReceived interrupt signal. Exiting...\n");
    exit(0);
}

/* Initialize and create the GStreamer pipeline */
static gboolean
create_pipeline(AppCtx *app_ctx)
{
    GstBus *bus = NULL;
    GstPad *osd_sink_pad = NULL;
    
    /* Create pipeline elements */
    app_ctx->pipeline = gst_pipeline_new("rfdetr-pipeline");
    app_ctx->source = gst_element_factory_make("v4l2src", "camera-source");
    app_ctx->streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    app_ctx->pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    app_ctx->nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    app_ctx->nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    app_ctx->transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
    app_ctx->sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    
    if (!app_ctx->pipeline || !app_ctx->source || !app_ctx->streammux || 
        !app_ctx->pgie || !app_ctx->nvvidconv || !app_ctx->nvosd || 
        !app_ctx->transform || !app_ctx->sink) {
        g_printerr("One or more elements could not be created. Exiting.\n");
        return FALSE;
    }
    
    /* Set element properties */
    g_object_set(G_OBJECT(app_ctx->source), "device", "/dev/video0", NULL);
    g_object_set(G_OBJECT(app_ctx->streammux), "width", 1920, "height", 1080,
                 "batch-size", 1, "batched-push-timeout", 4000000, NULL);
    g_object_set(G_OBJECT(app_ctx->pgie), 
                 "config-file-path", "config_infer_primary.txt", NULL);
    g_object_set(G_OBJECT(app_ctx->sink), "sync", FALSE, NULL);
    
    /* Add bus watch */
    bus = gst_pipeline_get_bus(GST_PIPELINE(app_ctx->pipeline));
    app_ctx->bus_watch_id = gst_bus_add_watch(bus, bus_call, app_ctx->loop);
    gst_object_unref(bus);
    
    /* Add elements to pipeline */
    gst_bin_add_many(GST_BIN(app_ctx->pipeline),
                     app_ctx->source, app_ctx->streammux, app_ctx->pgie,
                     app_ctx->nvvidconv, app_ctx->nvosd, app_ctx->transform,
                     app_ctx->sink, NULL);
    
    /* Link elements */
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    
    sinkpad = gst_element_get_request_pad(app_ctx->streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr("Streammux request sink pad failed. Exiting.\n");
        return FALSE;
    }
    
    srcpad = gst_element_get_static_pad(app_ctx->source, "src");
    if (!srcpad) {
        g_printerr("Failed to get src pad of source. Exiting.\n");
        return FALSE;
    }
    
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link source to stream muxer. Exiting.\n");
        return FALSE;
    }
    
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);
    
    /* Link the rest of the pipeline */
    if (!gst_element_link_many(app_ctx->streammux, app_ctx->pgie,
                               app_ctx->nvvidconv, app_ctx->nvosd,
                               app_ctx->transform, app_ctx->sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return FALSE;
    }
    
    /* Add probe to get informed of the meta data generated */
    osd_sink_pad = gst_element_get_static_pad(app_ctx->nvosd, "sink");
    if (!osd_sink_pad) {
        g_printerr("Unable to get sink pad\n");
        return FALSE;
    }
    
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);
    
    return TRUE;
}

/* Main function */
int
main(int argc, char *argv[])
{
    AppCtx app_ctx;
    GstStateChangeReturn ret;
    
    /* Initialize GStreamer */
    gst_init(&argc, &argv);
    
    /* Initialize app context */
    memset(&app_ctx, 0, sizeof(app_ctx));
    app_ctx.loop = g_main_loop_new(NULL, FALSE);
    
    /* Set up signal handler */
    signal(SIGINT, sig_handler);
    
    /* Create pipeline */
    if (!create_pipeline(&app_ctx)) {
        g_printerr("Failed to create pipeline. Exiting.\n");
        return -1;
    }
    
    /* Print pipeline info */
    g_print("RF-DETR DeepStream Application Starting...\n");
    g_print("Using camera: /dev/video0\n");
    g_print("ONNX Model: config_infer_primary.txt\n");
    g_print("Press Ctrl+C to stop\n\n");
    
    /* Start playing */
    ret = gst_element_set_state(app_ctx.pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Unable to set the pipeline to the playing state. Exiting.\n");
        return -1;
    }
    
    /* Run the main loop */
    g_print("Running...\n");
    g_main_loop_run(app_ctx.loop);
    
    /* Clean up */
    g_print("Cleaning up...\n");
    gst_element_set_state(app_ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(app_ctx.pipeline));
    g_source_remove(app_ctx.bus_watch_id);
    g_main_loop_unref(app_ctx.loop);
    
    g_print("Application ended successfully.\n");
    return 0;
}