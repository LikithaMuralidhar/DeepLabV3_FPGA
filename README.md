# DeepLabV3_FPGA
1. Download the Dataset
Download the PASCAL VOC 2012 training and validation dataset:
bashwget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
2. Extract the Dataset
Move the downloaded tar file to your dataset directory and extract it:
bashtar -xvf VOCtrainval_11-May-2012.tar
This will create a directory structure containing the VOC dataset.
3. Install Dependencies
Install all required Python packages:
bashpip install -r requirements.txt
Usage
Training the Model
To train the model with the prepared dataset:
bashpython3 train.py
Model Quantization
After training is complete, quantize the model for optimized inference:
bashpython3 Quantization.py

5. Quantaize the model
   python3 Quantization.py

   
