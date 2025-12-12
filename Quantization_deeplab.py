import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

IMAGE_SIZE = 512
BATCH_SIZE = 8


FLOAT_MODEL_PATH = "deeplabv3_best.h5"


CALIB_DIR = "calib_images"


OUTPUT_H5 = "deeplabv3_ptq.h5"


def create_calib_dataset(calib_dir, image_size, batch_size):
    files = [
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        raise RuntimeError(f"No images found in {calib_dir}")

    def gen():
        for path in files:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0
            yield img

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(
            shape=(image_size, image_size, 3), dtype=tf.float32
        ),
    )
    ds = ds.batch(batch_size)
    return ds


def main():
    print("Loading float model from:", FLOAT_MODEL_PATH)
    float_model = tf.keras.models.load_model(FLOAT_MODEL_PATH, compile=False)

    print("Building calibration dataset from:", CALIB_DIR)
    calib_ds = create_calib_dataset(CALIB_DIR, IMAGE_SIZE, BATCH_SIZE)

    print("Creating VitisQuantizer...")
    quantizer = vitis_quantize.VitisQuantizer(float_model)

    print("Running PTQ quantization...")
    quantized_model = quantizer.quantize_model(
        calib_dataset=calib_ds,
        calib_steps=50,
        calib_batch_size=BATCH_SIZE,
    )

    print("Saving quantized model as Keras H5 ->", OUTPUT_H5)
    quantized_model.save(OUTPUT_H5)  

    print("Done! Quantized Keras model ready:", OUTPUT_H5)


if __name__ == "__main__":
    main()

