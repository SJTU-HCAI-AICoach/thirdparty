from typing import List, Tuple
import os
import numpy as np
import tensorrt as trt
import cv2 as cv
import time
import ctypes

ctypes.CDLL("libmmdeploy_tensorrt_ops.so")

def build_model(onnx_file_path):
    engine_file_path = onnx_file_path.replace('.onnx', '_dynamic.engine')
    build_engine(onnx_file_path, engine_file_path, True)

def build_engine(onnx_file_path, engine_file_path, half=True):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # vitpose
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1,3,256,192), (32,3,256,192), (64,3,256,192)) 
    config.add_optimization_profile(profile)
    # yolo
    # profile = builder.create_optimization_profile()
    # profile.set_shape('input', (1,3,416,416), (1,3,416,416), (1,3,416,416)) 
    # config.add_optimization_profile(profile)
    # simcc
    # profile = builder.create_optimization_profile()
    # profile.set_shape('input', (1,3,384,288), (1,3,384,288), (1,3,384,288)) 
    # config.add_optimization_profile(profile)

    #reid
    #profile = builder.create_optimization_profile()
    #profile.set_shape('base_images', (1,3,256,128), (32,3,256,128), (64,3,256,128)) 
    #config.add_optimization_profile(profile)


    # config.max_workspace_size = 4 * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file_path}')
    half &= builder.platform_has_fast_fp16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_serialized_network(network, config) as engine, open(engine_file_path, 'wb') as f:
        f.write(engine)
    # with builder.build_engine(network, config) as engine, open(engine_file_path, 'wb') as t:
    #     t.write(engine.serialize())
    return engine_file_path
build_model("ckpt/vitpose-b-multi-coco.onnx")
# build_model("ckpt/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx")
# build_model("ckpt/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx")
# build_model("ckpt/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx")

# from rtmlib.tools.object_detection.yolox import YOLOX
# det_model = YOLOX(
#     # accurate model
#     "ckpt/yolox-x_fp16_dynamic.onnx",
#     model_input_size=(640, 640),
#     # tiny model
#     backend="tensorrt",
#     score_thr=0.85,
#     device="cuda",
# )
# img2 = cv.imread("test.jpg")
# #img = cv.imread("/home/lab4dv/Desktop/freecap/dep/mmdeploy/demo/resources/human-pose.jpg")
# res = det_model(img2)
# print(res)
# # cv.imwrite("test.jpg", img2)
# # cv.imshow("img2", img2)
# # cv.waitKey(0)
# # print(res)



