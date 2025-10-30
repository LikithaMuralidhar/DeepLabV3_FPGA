# DeepLabV3_FPGA
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

4.Training the Model
To train the model with the prepared dataset:

```bash
python3 train.py
```
5. Model Quantization
After training is complete, quantize the model for optimized inference:
```bash
python3 Quantization.py
```



   
