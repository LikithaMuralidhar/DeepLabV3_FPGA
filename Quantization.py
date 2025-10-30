
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

print("TensorFlow version:", tf.__version__)

# Configuration
IMAGE_SIZE = 512
NUM_CLASSES = 21
VOC_PATH = '/workspace/dataset/VOCdevkit/VOC2012'
MODEL_PATH = 'deeplabv3_best.h5'  
QUANTIZED_MODEL_PATH = 'deeplabv3_quantized.h5'


VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
], dtype=np.uint8)

def decode_segmentation_mask(mask):
   
    h, w = mask.shape[:2]
    output = np.zeros((h, w), dtype=np.uint8)
    
    for class_idx, color in enumerate(VOC_COLORMAP):
        matches = np.all(mask == color, axis=-1)
        output[matches] = class_idx
    
    boundary = np.all(mask >= 224, axis=-1)
    output[boundary] = 0
    
    return output

def load_calibration_data(voc_path, num_samples=100):
   
    
    val_file = os.path.join(voc_path, "ImageSets/Segmentation/val.txt")
    with open(val_file, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()][:num_samples]
    
    image_dir = os.path.join(voc_path, "JPEGImages")
    
    calibration_data = []
    
    for idx, image_id in enumerate(val_ids):
        if idx % 10 == 0:
            print(f"  Loading {idx}/{num_samples}...")
        
        try:
            img_path = os.path.join(image_dir, f"{image_id}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            calibration_data.append(img)
        except Exception as e:
            print(f"Error loading {image_id}: {e}")
            continue
    
   
    return np.array(calibration_data, dtype=np.float32)

def representative_dataset_gen(calibration_data, batch_size=1):
    
    num_samples = len(calibration_data)
    for i in range(0, num_samples, batch_size):
        batch = calibration_data[i:i+batch_size]
        yield [batch.astype(np.float32)]

def quantize_model_tflite(model_path, calibration_data, output_path):
  
   
    # Load model
   
    model = keras.models.load_model(model_path)
    
    
    # Create converter
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Provide calibration data
    def representative_dataset():
        for data in calibration_data:
            yield [np.expand_dims(data, axis=0).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Force INT8 for all ops (if possible)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    
    # Convert
    print("\nQuantizing to INT8...")
    try:
        tflite_model = converter.convert()
        
        # Save
        tflite_path = output_path.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
    
        size_mb = len(tflite_model) / (1024 * 1024)
    
        
        return tflite_path
        
    except Exception as e:
        print(f"\n TFLite quantization failed: {e}")
       
        return None

def save_for_vitis_ai(model_path, output_dir='model_for_vitis'):
  

    os.makedirs(output_dir, exist_ok=True)
    
   
    model = keras.models.load_model(model_path)
    
   
    savedmodel_path = os.path.join(output_dir, 'float_model')
    
    tf.saved_model.save(model, savedmodel_path)
  
    
    
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        
        # Convert Keras model to concrete function
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )
        
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        
        # Save frozen graph
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=output_dir,
            name='frozen_graph.pb',
            as_text=False
        )
       
        
    except Exception as e:
        print(f"Warning: Could not save frozen graph: {e}")
    

    
    return savedmodel_path

def compare_accuracy(original_model_path, quantized_model_path, test_data, test_labels):
   
   
    print("Accuracy Comparison")
 
   
    print("Loading original float32 model...")
    float_model = keras.models.load_model(original_model_path)
    
  
    print("Testing float32 model...")
    float_pred = float_model.predict(test_data, verbose=0)
    float_pred = np.argmax(float_pred, axis=-1)
    

    float_acc = np.mean(float_pred == test_labels)
    
    print(f"\nFloat32 Accuracy: {float_acc*100:.2f}%")
    
   
    
    return float_acc

def main():
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n Model not found: {MODEL_PATH}")
        print("Please train the model first!")
        return

  
    calibration_data = load_calibration_data(VOC_PATH, num_samples=100)
    
   
    tflite_path = quantize_model_tflite(MODEL_PATH, calibration_data, QUANTIZED_MODEL_PATH)
    
    
    savedmodel_path = save_for_vitis_ai(MODEL_PATH)
    

    print("Quantization Complete!")
   
    


if __name__ == "__main__":
    main()
