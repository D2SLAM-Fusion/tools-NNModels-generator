## SuperPoint_AND_NetVLAD
### requirement
1. tensorrt /usr/src/tensorrt/bin/trtexec 

## HITNET Generator

This repo can transfer HINET.pb trained under ETH3d dataset to ONNX model with different input size and float size

### Requirement
1. openvino2tensorflow 
2. tensorflow_datasets (pip install )

run `save_model_to_onnx.sh` directly and modify H W C value to get ONNX models

this method comes from [script](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/142_HITNET/convert_script.txt)![image-20230917215307625](https://raw.githubusercontent.com/Peize-Liu/my-images/master/202309172156634.png)

download model from [model](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/142_HITNET/download.sh)