import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def grad_cam(model, img, layer_name="conv5_block3_out", label_name=None, category_id=None):
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        predictions = tf.convert_to_tensor(predictions)  # Ensure predictions is a tensor

        # Print predictions shape for debugging
        print("Predictions shape:", predictions.shape)

        # Handle different output shapes
        predictions = tf.squeeze(predictions)  # Remove single-dimensional entries
        if len(predictions.shape) == 1:  # Single-dimension output
            category_id = 0  # Default to the first output, assuming binary classification
        else:
            if category_id is None:
                category_id = np.argmax(predictions)  # Use the predicted class if not specified
            if category_id >= predictions.shape[0]:  # Check if category_id is valid
                raise ValueError(f"category_id {category_id} is out of bounds for predictions shape {predictions.shape}")
            if label_name is not None:
                print(label_name[category_id])
        
        output = predictions[category_id]  # Directly access the prediction
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.reduce_max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap.numpy())


# Grad-CAM++
def grad_cam_plus(model, img, layer_name="conv5_block3_out", label_name=None, category_id=None):
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                predictions = tf.convert_to_tensor(predictions)

                # Print predictions shape for debugging
                print("Predictions shape:", predictions.shape)

                # Handle predictions shape
                predictions = tf.squeeze(predictions)  # Remove singleton dimensions
                if category_id is None:
                    category_id = np.argmax(predictions)  # Use the predicted class if not specified
                
                # Check if category_id is within the valid range
                if category_id < 0 or category_id >= predictions.shape[0]:
                    raise ValueError(f"category_id {category_id} is out of bounds for predictions shape {predictions.shape}")

                if label_name is not None:
                    print(label_name[category_id])
                
                output = predictions[category_id]  # Access the correct prediction
                
                # Gradients calculation
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = tf.reduce_sum(conv_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    weights = tf.maximum(conv_first_grad[0], 0.0)
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 1))
    grad_cam_map = tf.reduce_sum(deep_linearization_weights * conv_output[0], axis=2)

    heatmap = tf.maximum(grad_cam_map, 0)
    max_heat = tf.reduce_max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap.numpy()
