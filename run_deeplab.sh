#!/bin/bash
# compile_for_v70.sh

echo "Compiling DeepLabV3 for Alveo V70..."

# Activate Vitis AI environment
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-tensorflow2

# Compile for V70
vai_c_tensorflow2 \
    --model deeplabv3_savedmodel \
    --arch /opt/vitis_ai/compiler/arch/DPUCV2DX8G/V70/arch.json \
    --output_dir compiled_model \
    --net_name deeplabv3_v70

echo "✓ Compilation complete!"
echo "XModel location: compiled_model/deeplabv3_v70.xmodel"
