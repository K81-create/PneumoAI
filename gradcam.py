"""
gradcam.py — Grad-CAM (Gradient-weighted Class Activation Mapping)

Generates visual explanations for CNN predictions by highlighting the
regions of the input image that most influenced the model's decision.

Uses TensorFlow GradientTape to compute gradients of the predicted class
score with respect to the feature maps of the last convolutional layer,
then overlays a colour heatmap on the original X-ray image.
"""

import tensorflow as tf
import numpy as np
import cv2


def get_last_conv_layer(model):
    """
    Automatically detect the last Conv2D layer in the model.
    Iterates backwards through layers and returns the first Conv2D found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(model, img_array, img_path, save_path):
    """
    Generate a Grad-CAM heatmap and overlay it on the original image.

    Parameters:
        model      – Trained Keras model
        img_array  – Preprocessed image array (1, 128, 128, 3), normalised to [0,1]
        img_path   – Path to the original image on disk (for overlay)
        save_path  – Where to save the resulting Grad-CAM image

    Returns:
        save_path on success, or None if generation fails.
    """
    try:
        # -----------------------------------------------------------------
        # Step 1: Identify the last convolutional layer
        # -----------------------------------------------------------------
        last_conv_layer_name = get_last_conv_layer(model)

        if last_conv_layer_name is None:
            print("[Grad-CAM] No Conv2D layer found in the model.")
            return None

        last_conv_layer = model.get_layer(last_conv_layer_name)

        # -----------------------------------------------------------------
        # Step 2: Build a sub-model that outputs the conv layer activations.
        #         For Sequential models in newer Keras, model.output may
        #         not be available, so we manually forward through the
        #         remaining layers after the target conv layer.
        # -----------------------------------------------------------------
        conv_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=last_conv_layer.output
        )

        # Identify layers that come AFTER the target conv layer
        remaining_layers = []
        found_conv = False
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                found_conv = True
                continue
            if found_conv:
                remaining_layers.append(layer)

        # -----------------------------------------------------------------
        # Step 3: Compute gradients of the prediction w.r.t. the conv
        #         feature maps using GradientTape
        # -----------------------------------------------------------------
        input_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            # Get conv layer output
            conv_outputs = conv_model(input_tensor)
            tape.watch(conv_outputs)

            # Forward through remaining layers to get predictions
            x = conv_outputs
            for layer in remaining_layers:
                x = layer(x)
            predictions = x

            # For binary classification with sigmoid, use the single output
            loss = predictions[:, 0]

        # Gradients of the output neuron w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[Grad-CAM] Gradients could not be computed (None).")
            return None

        # -----------------------------------------------------------------
        # Step 4: Pool the gradients across spatial dimensions to get
        #         per-channel importance weights
        # -----------------------------------------------------------------
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # -----------------------------------------------------------------
        # Step 5: Weight each channel by its importance and compute
        #         the heatmap as a spatial average
        # -----------------------------------------------------------------
        conv_outputs = conv_outputs[0]  # Remove batch dimension

        # Multiply each channel by its pooled gradient weight
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU — we only care about features that have a positive
        # influence on the predicted class
        heatmap = np.maximum(heatmap.numpy(), 0)

        # Normalise to [0, 1]
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)

        # -----------------------------------------------------------------
        # Step 6: Resize heatmap to match the original image dimensions
        # -----------------------------------------------------------------
        original_img = cv2.imread(img_path)

        if original_img is None:
            print(f"[Grad-CAM] Could not read image: {img_path}")
            return None

        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Convert to 8-bit colour-mapped heatmap (JET colourmap)
        heatmap_coloured = np.uint8(255 * heatmap_resized)
        heatmap_coloured = cv2.applyColorMap(heatmap_coloured, cv2.COLORMAP_JET)

        # -----------------------------------------------------------------
        # Step 7: Overlay heatmap on the original image
        # -----------------------------------------------------------------
        superimposed = cv2.addWeighted(
            original_img, 0.6,
            heatmap_coloured, 0.4,
            0
        )

        # -----------------------------------------------------------------
        # Step 8: Save the Grad-CAM visualisation to disk
        # -----------------------------------------------------------------
        cv2.imwrite(save_path, superimposed)
        print(f"[Grad-CAM] Saved heatmap to {save_path}")

        return save_path

    except Exception as e:
        # Graceful error handling — prediction still works even if
        # Grad-CAM generation fails
        print(f"[Grad-CAM] Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None
