import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize


IMAGE_SIZE = 512
NUM_CLASSES = 21
BATCH_SIZE = 4
VOC_PATH = "/workspace/dataset/VOCdevkit/VOC2012"
FP32_MODEL_PATH = "deeplabv3_best.h5"   
QAT_OUTPUT_H5 = "deeplabv3_qat.h5"      
QAT_EPOCHS = 10


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


def load_data_batch(image_ids, voc_path, image_size):
    images = []
    masks = []

    image_dir = os.path.join(voc_path, "JPEGImages")
    mask_dir = os.path.join(voc_path, "SegmentationClass")

    for image_id in image_ids:
        try:
            img_path = os.path.join(image_dir, f"{image_id}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0

            mask_path = os.path.join(mask_dir, f"{image_id}.png")
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = decode_segmentation_mask(mask)
            mask = cv2.resize(mask, (image_size, image_size),
                              interpolation=cv2.INTER_NEAREST)

            images.append(img)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading {image_id}: {e}")
            continue

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)


def data_generator(image_ids, voc_path, image_size, batch_size, shuffle=True):
    num_samples = len(image_ids)
    while True:
        if shuffle:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_ids = [image_ids[i] for i in batch_indices]

            images, masks = load_data_batch(batch_ids, voc_path, image_size)
            if len(images) > 0:
                yield images, masks



def create_calib_dataset_from_ids(image_ids, voc_path, image_size, batch_size):
    image_dir = os.path.join(voc_path, "JPEGImages")

    def gen():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0
            yield img

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(
            shape=(image_size, image_size, 3),
            dtype=tf.float32
        ),
    )
    ds = ds.batch(batch_size)
    return ds



def evaluate_segmentation_model(model, val_ids, voc_path, image_size,
                                batch_size, num_classes, max_steps=None):
    val_gen = data_generator(val_ids, voc_path, image_size, batch_size, shuffle=False)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_pixels = 0
    correct_pixels = 0

    steps = len(val_ids) // batch_size
    if max_steps is not None:
        steps = min(steps, max_steps)

    for _ in range(steps):
        images, masks = next(val_gen)
        logits = model.predict(images, verbose=0)
        preds = np.argmax(logits, axis=-1).astype(np.int32)
        masks = masks.astype(np.int32)

        valid = (masks >= 0) & (masks < num_classes)
        for i in range(preds.shape[0]):
            p = preds[i][valid[i]]
            g = masks[i][valid[i]]
            total_pixels += g.size
            correct_pixels += np.sum(p == g)
            k = (g * num_classes + p).astype(np.int64)
            conf_mat += np.bincount(
                k, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    pixel_acc = correct_pixels / (total_pixels + 1e-7)
    inter = np.diag(conf_mat)
    union = conf_mat.sum(1) + conf_mat.sum(0) - inter
    iou = inter / (union + 1e-7)
    miou = np.nanmean(iou)
    return pixel_acc, miou


def main():
    
    train_file = os.path.join(VOC_PATH, "ImageSets/Segmentation/train.txt")
    val_file = os.path.join(VOC_PATH, "ImageSets/Segmentation/val.txt")

    with open(train_file, "r") as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(val_file, "r") as f:
        val_ids = [line.strip() for line in f.readlines()]

   
    print("Loading FP32 model:", FP32_MODEL_PATH)
    float_model = keras.models.load_model(FP32_MODEL_PATH, compile=False)

   
    calib_ids = train_ids[:200]  
    calib_ds = create_calib_dataset_from_ids(
        calib_ids, VOC_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE
    )

   
    quantizer = vitis_quantize.VitisQuantizer(float_model)

    print("Creating QAT model ")
    qat_model = quantizer.get_qat_model(
        init_quant=True,
        calib_dataset=calib_ds,
        calib_steps=50,
        train_with_bn=True,
    )

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

 
    train_gen = data_generator(train_ids, VOC_PATH, IMAGE_SIZE, BATCH_SIZE, shuffle=True)
    val_gen = data_generator(val_ids, VOC_PATH, IMAGE_SIZE, BATCH_SIZE, shuffle=False)

    steps_per_epoch = len(train_ids) // BATCH_SIZE
    val_steps = len(val_ids) // BATCH_SIZE

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "deeplabv3_qat_best.h5",
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor="val_loss",
            verbose=1,
            min_lr=1e-7,
        ),
        keras.callbacks.CSVLogger("training_log_qat.csv", append=False),
    ]

    print("\n=== Starting QAT fine-tuning ===")
    qat_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=QAT_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    
    print("\nEvaluating QAT model on validation set (small subset)…")
    pixel_acc, miou = evaluate_segmentation_model(
        qat_model, val_ids, VOC_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, max_steps=50
    )
    print(f"QAT Pixel Accuracy: {pixel_acc:.4f}")
    print(f"QAT mIoU         : {miou:.4f}")

  
    print("\nExporting deployable QAT model...")
    deploy_model = quantizer.get_deploy_model(qat_model)
    deploy_model.save(QAT_OUTPUT_H5)  

    print(f"\nQAT model saved as: {QAT_OUTPUT_H5}")


if __name__ == "__main__":
    main()

