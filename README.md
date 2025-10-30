# DeepLabV3_FPGA

1. Download the dataset
   $ wget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar

    2. Move the tar file to dataset directory
       
       $tar -xvf [filename] # to unzip the folder

3. Install all the dependencies
    pip install -r requirements.txt

4. train the model
   python3 train.py

5. Quantaize the model
   python3 Quantization.py

   
