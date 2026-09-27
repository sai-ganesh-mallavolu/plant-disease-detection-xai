import tensorflow as tf
import numpy as np
import cv2


# ==========================================
# Create Grad-CAM Components
# ==========================================

def create_gradcam_components(model):

    # Find the MobileNetV2 base model
    # by looking for the "out_relu" layer
    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            try:
                layer.get_layer("out_relu")
                base_model = layer
                break

            except ValueError:
                continue

    if base_model is None:
        raise ValueError(
            "MobileNetV2 base model with out_relu layer not found."
        )

    # Same Grad-CAM model as the notebook
    gradcam_base = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("out_relu").output
    )

    # Same classification head as the notebook
    gap_layer = model.get_layer(
        "global_average_pooling2d"
    )

    dense_layer = model.get_layer(
        "dense"
    )

    final_layer = model.layers[-1]

    return (
        gradcam_base,
        gap_layer,
        dense_layer,
        final_layer
    )


# ==========================================
# Generate Grad-CAM
# ==========================================

def generate_gradcam(
    img_array,
    predicted_index,
    gradcam_base,
    gap_layer,
    dense_layer,
    final_layer
):

    img_array = tf.cast(
        img_array,
        tf.float32
    )

    # MobileNetV2 preprocessing
    img_preprocessed = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )
    )

    with tf.GradientTape() as tape:

        # Get MobileNetV2 feature maps
        conv_outputs = gradcam_base(
            img_preprocessed,
            training=False
        )

        # Watch feature maps
        tape.watch(conv_outputs)

        # Classification head
        x = gap_layer(conv_outputs)

        x = dense_layer(x)

        predictions = final_layer(x)

        # Score of predicted class
        class_score = predictions[
            :,
            predicted_index
        ]

        # Calculate gradients
        grads = tape.gradient(
            class_score,
            conv_outputs
        )

        # Average gradients
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        # Remove batch dimension
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads[0]

        # Weighted feature maps
        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        # Remove negative values
        heatmap = tf.maximum(
            heatmap,
            0
        )

        # Normalize
        heatmap = heatmap / (
            tf.reduce_max(heatmap)
            + tf.keras.backend.epsilon()
        )

        return heatmap.numpy()


# ==========================================
# Create Grad-CAM Overlay
# ==========================================

def create_gradcam_overlay(
    image_path,
    heatmap
):

    # Read original image
    original = cv2.imread(
        image_path
    )

    if original is None:
        raise ValueError(
            "Unable to read image."
        )

    # Convert BGR to RGB
    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Resize Grad-CAM heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply color map
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay heatmap with original image
    superimposed = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return (
        original,
        heatmap_color,
        superimposed
    )