# DeepLabV3_FPGA
This project aims to accelerate the DeepLabV3 semantic segmentation model by deploying it on a Xilinx Alveo U280 FPGA using the Vitis AI framework. DeepLabV3 algorithm will be trained in Keras using benchmark datasets such as Pascal VOC 2012. The trained floating-point model will be converted into a TensorFlow graph, frozen, and quantized from 32-bit floating-point to 8-bit fixed-point representation. The quantized model will then be compiled into an XModel and deployed on the FPGA accelerator. Quantization-aware training and graph-level optimizations will be applied to minimize accuracy loss while improving computational efficiency. The performance will be evaluated in terms of segmentation, accuracy, inference latency, and  throughput.

While setting up the V70 in Cloud lab , make sure to select the docker image tensorflow2

1. Download the Dataset
Download the PASCAL VOC 2012 training and validation dataset:

```bash
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
python3 train.py
```
5. Model Quantization
After training is complete, quantize the model for optimized inference:
```bash
python3 Quantization.py
```



   
