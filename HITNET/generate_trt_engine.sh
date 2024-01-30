current_dir=$(pwd)

# /usr/src/tensorrt/bin/trtexec --onnx=${current_dir}/hitnet_saved_model_1x240x320/hitnet_1x240x320_model_float32_opt.onnx \
#   --saveEngine=${current_dir}/hitnet_saved_model_1x240x320/hitnet_1x240x320_model_float32_opt.trt  --fp16

  /usr/src/tensorrt/bin/trtexec --onnx=${current_dir}/hitnet_saved_model_1x240x320/hitnet_1x240x320_model_float16_quant_opt.onnx \
  --saveEngine=${current_dir}/hitnet_saved_model_1x240x320/hitnet_1x240x320_model_float16_quant_opt.trt  --best