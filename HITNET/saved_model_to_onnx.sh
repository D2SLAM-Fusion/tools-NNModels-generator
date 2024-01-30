#/bin/bash
H=240
W=320
C=1

current_dir=$(pwd)

#save_model_to_tflite comes from openvino2tensorflow
saved_model_to_tflite \
--saved_model_dir_path ${current_dir}/saved_model \
--input_shapes [${C},${H},${W},2] \
--model_output_dir_path ${current_dir}/hitnet_saved_model_${C}x${H}x${W} \
--output_no_quant_float32_tflite \
--output_weight_quant_tflite \
--output_float16_quant_tflite

python3 -m tf2onnx.convert \
--opset 12 \
--inputs-as-nchw input \
--tflite ${current_dir}/hitnet_saved_model_${C}x${H}x${W}/model_float32.tflite \
--output ${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float32.onnx

python3 -m onnxsim \
${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float32.onnx \
${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float32_opt.onnx

python3 -m tf2onnx.convert \
--opset 12 \
--inputs-as-nchw input \
--tflite ${current_dir}/hitnet_saved_model_${C}x${H}x${W}/model_float16_quant.tflite \
--output ${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float16_quant.onnx


python3 -m onnxsim \
${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float16_quant.onnx \
${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float16_quant_opt.onnx

# generate trt engine
/usr/src/tensorrt/bin/trtexec --onnx=${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float32_opt.onnx --saveEngine=${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float32_opt.trt  --fp16
/usr/src/tensorrt/bin/trtexec --onnx=${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float16_quant_opt.onnx --saveEngine=${current_dir}/hitnet_saved_model_${C}x${H}x${W}/hitnet_${C}x${H}x${W}_model_float16_quant_opt.trt  --fp16