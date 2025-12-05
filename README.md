# DeepLabV3_FPGA
This project aims to accelerate the DeepLabV3 semantic segmentation model by deploying it on a Xilinx Alveo U280 FPGA using the Vitis AI framework. DeepLabV3 algorithm will be trained in Keras using benchmark datasets such as Pascal VOC 2012. The trained floating-point model will be quantized from 32-bit floating-point to 8-bit integer representation. The quantized model will then be compiled into an XModel and deployed on the FPGA accelerator. Quantization-aware training and Post training Quantization will be applied to minimize accuracy loss while improving computational efficiency. The performance will be evaluated in terms of segmentation, accuracy, inference latency, and  throughput.

While setting up the V70 in Cloud lab , make sure to select the docker image tensorflow2

Prerequisites
Clone VITIS-AI and V70 setup
```bash
git clone https://github.com/Xilinx/Vitis-AI
cd Vitis-AI/board_setup/v70 (Make sure to Include correct file path)
source ./install.sh
```
Pull the lastest Docker image and also set the required environment variables for the DPU
```bash
cd <Vitis-AI install path>/Vitis-AI/
./docker_run.sh xilinx/vitis-ai-tensorflow2-cpu:latest
source /workspace/board_setup/v70/setup.sh DPUCV2DX8G_v70
```
Project

1. Download the Dataset
Download the PASCAL VOC 2012 training and validation dataset:

```bash
mkdir dataset
cd dataset
wget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
```
2. Extract the Dataset
Move the downloaded tar file to your dataset directory and extract it:

```bash
tar -xvf VOCtrainval_11-May-2012.tar
```
This will create a directory structure containing the VOC dataset.

3. Install Dependencies
Install all required Python packages:

```bash
pip install -r requirements.txt
```

4. Training the Model
To train the model with the prepared dataset:

```bash
python3 Train.py
```
After this deeplabv3_best.h5 and deeplabv3_final.h5 files will be generated
Run deeplab.py to get the Inference of Base float 32 nit precesion

5. Quantization - PQT(Post Training Quantization)
```bash
python3 Quantization_deeplab.py
```
deeplabv3_ptq.h5 file will be generated

6. Quantization - QAT(Quantization Aware Training)
```bash
python3 deeplab_qat.py
```
deeplabv3_qat.h5 file will be generated.

6. Vitis AI compilation
```bash
vai_c_tensorflow2 \
  --model deeplabv3_ptq.h5 \
  --arch /opt/vitis_ai/compiler/arch/DPUCV2DX8G/V70/arch.json \
  --output_dir compiled_v70_ptq \
  --net_name deeplabv3_ptq

vai_c_tensorflow2 \
  --model deeplabv3_qat.h5 \
  --arch /opt/vitis_ai/compiler/arch/DPUCV2DX8G/V70/arch.json \
  --output_dir compiled_v70_ptq \
  --net_name deeplabv3_qat
```
.xmodel file will be generated accordingly

7. Inference from Xmodel
 ```bash
    python3 run_deeplab.py compiled_v70_ptq/deploy.xmodel
   ```
    #make sure you change the file name accordingly
      



   
