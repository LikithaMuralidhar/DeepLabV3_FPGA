import time
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 512
MODEL_PATH = "deeplabv3_best.h5"   


import cv2
import glob

def load_voc_batch(batch_size, image_size=512):
    
    image_paths = glob.glob("dataset/VOCdevkit/VOC2012/JPEGImages/*.jpg")

    batch = []
    for i in range(batch_size):
        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32) / 255.0
        batch.append(img)

    return np.array(batch) 

def benchmark_fp32(model_path, batch_size=1, num_warmup=10, num_iters=50):
    print(f"Loading FP32 model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    
    dummy_input = load_voc_batch(batch_size, IMAGE_SIZE)

  
   
    def infer(x):
        return model(x, training=False)

    # Warmup
    
    for _ in range(num_warmup):
        _ = infer(dummy_input)

    # Timed run
    print(f"Running {num_iters} timed iterations...")
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = infer(dummy_input)
    end = time.perf_counter()

    total_time = end - start
    avg_batch_time = total_time / num_iters
    latency_ms = (avg_batch_time / batch_size) * 1000.0
    throughput = (batch_size * num_iters) / total_time

    print("\n=== FP32 Keras Benchmark (no quantization) ===")
    print(f"Latency   : {latency_ms:.3f} ms / image")
    print(f"Throughput: {throughput:.2f} images/s")

    return latency_ms, throughput


if __name__ == "__main__":
    benchmark_fp32(MODEL_PATH, batch_size=4, num_warmup=10, num_iters=50)

