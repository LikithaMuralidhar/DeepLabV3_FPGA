# DeepLabV3_FPGA
1. Download the Dataset
Download the PASCAL VOC 2012 training and validation dataset:
`inline code`
```bash
wget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
```
2. Extract the Dataset
Move the downloaded tar file to your dataset directory and extract it:
`inline code`
```bash
tar -xvf VOCtrainval_11-May-2012.tar
```
This will create a directory structure containing the VOC dataset.
4. Install Dependencies
Install all required Python packages:
`inline code`
```bash
pip install -r requirements.txt
```
#Usage
Training the Model
To train the model with the prepared dataset:
`inline code`
```bash
python3 train.py
```

Model Quantization
After training is complete, quantize the model for optimized inference:
`inline code`
```bash
python3 Quantization.py
```



   
