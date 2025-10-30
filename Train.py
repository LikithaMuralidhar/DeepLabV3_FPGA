
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# Configuration
IMAGE_SIZE = 512
NUM_CLASSES = 21  
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001
VOC_PATH = '/workspace/dataset/VOCdevkit/VOC2012'


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

def convolution_block(block_input, num_filters=256, kernel_size=3, 
                      dilation_rate=1, use_bias=False):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    
    x = layers.GlobalAveragePooling2D(keepdims=True)(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    x = layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([x, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size=512, num_classes=21):
    model_input = keras.Input(shape=(image_size, image_size, 3), name='input_image')
    
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", 
        include_top=False, 
        input_tensor=model_input
    )
    
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    model_output = layers.Conv2D(
        num_classes, 
        kernel_size=(1, 1), 
        padding="same",
        name='output'
    )(x)
    
    return keras.Model(inputs=model_input, outputs=model_output, name='deeplabv3plus')

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
        
        # Go through all batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_ids = [image_ids[i] for i in batch_indices]
            
            # Load batch
            images, masks = load_data_batch(batch_ids, voc_path, image_size)
            
           
            if len(images) > 0:
                yield images, masks
        

def main():
    
    train_file = os.path.join(VOC_PATH, "ImageSets/Segmentation/train.txt")
    val_file = os.path.join(VOC_PATH, "ImageSets/Segmentation/val.txt")
    
    with open(train_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    with open(val_file, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    #print(f"Training samples: {len(train_ids)}")
    #print(f"Validation samples: {len(val_ids)}")
    
    # Create data generators
    train_gen = data_generator(train_ids, VOC_PATH, IMAGE_SIZE, BATCH_SIZE, shuffle=True)
    val_gen = data_generator(val_ids, VOC_PATH, IMAGE_SIZE, BATCH_SIZE, shuffle=False)
    
    # Calculate steps
    steps_per_epoch = len(train_ids) // BATCH_SIZE
    validation_steps = len(val_ids) // BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Load existing model if available (continue training)
    if os.path.exists('deeplabv3_best.h5'):
        
        model = keras.models.load_model('deeplabv3_best.h5')
       
    else:
        # Create new model
        print("\nBuilding new model...")
        model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        
        print("Model built successfully!")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'deeplabv3_best.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            monitor='val_loss',
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.CSVLogger('training_log.csv', append=True),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    
   
    print("Starting Training...")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    
    
    model.save('deeplabv3_final.h5')
    
    try:
        tf.saved_model.save(model, 'deeplabv3_savedmodel')
    except Exception as e:
        print(f"Warning: Could not save SavedModel format: {e}")
    
  

if __name__ == "__main__":
    main()
