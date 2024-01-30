#!/bin/bash

H=150
W=300
C=1
current_dir=$(pwd)

/usr/src/tensorrt/bin/trtexec --onnx=superpoint_v1_dyn_size.onnx --saveEngine=superpoint_v1_dyn_size_onnx_${H}_${W}.trt --shapes=image:1x1x${H}x${W} --best
/usr/src/tensorrt/bin/trtexec --onnx=mobilenetvlad_dyn_size.onnx --saveEngine=mobilenetvlad_dyn_size_onnx_${H}_${W}.trt --shapes=image:0:1x${H}x${W}x1 --best

# test inference
# /usr/src/tensorrt/bin/trtexec --loadEngine=superpoint_v1_dyn_size_onnx_${H}_${W}.trt --streams=4 --iterations=100
# /usr/src/tensorrt/bin/trtexec --loadEngine=mobilenetvlad_dyn_size_onnx_${H}_${W}.trt --streams=4 --iterations=100 
