# create_pretrained_model.py
"""
Create DeepLabV3 model with ImageNet pre-trained backbone
No VOC dataset needed - ready for FPGA deployment
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

IMAGE_SIZE = 512
NUM_CLASSES = 21

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
    """DeepLabV3+ with ImageNet pre-trained ResNet50 backbone"""
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    # ResNet50 with ImageNet weights (pre-trained!)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet",  # Automatically downloads ImageNet weights
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
        padding="same"
    )(x)
    
    return keras.Model(inputs=model_input, outputs=model_output)

# Create model with pre-trained weights
print("Creating DeepLabV3+ with ImageNet pre-trained ResNet50...")
model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

print("\n✓ Model created with pre-trained backbone!")
print("  - ResNet50 backbone: ImageNet weights")
print("  - Segmentation head: Random initialization")
print("  - Ready for FPGA deployment testing")

# Compile (optional - mainly for format compatibility)
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Save for FPGA deployment
model.save('deeplabv3_pretrained.h5')
print("\n✓ Model saved: deeplabv3_pretrained.h5")

# Also save as SavedModel
tf.saved_model.save(model, 'deeplabv3_savedmodel')
print("✓ SavedModel saved: deeplabv3_savedmodel/")

# Test inference
print("\nTesting inference...")
test_input = np.random.rand(1, 512, 512, 3).astype(np.float32)
output = model.predict(test_input, verbose=0)
print(f"✓ Inference successful!")
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {output.shape}")

print("\n" + "="*60)
print("READY FOR FPGA DEPLOYMENT!")
print("Next steps:")
print("1. Quantize this model")
print("2. Compile for Alveo V70")
print("3. Deploy and benchmark")
print("="*60)
