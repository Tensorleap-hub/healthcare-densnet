import os
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

from leap_binder import *


def check_custom_integration():
    preprocess = preprocess_func()
    train = preprocess[0]
    val = preprocess[1]
    test = preprocess[2]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/DenseNet201.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    set = train
    for idx in range(20):
        # get input and gt
        image_input = input_encoder(idx, set)
        concat = np.expand_dims(image_input, axis=0)
        gt = gt_encoder(idx, set)
        y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

        y_pred = cnn([concat])

        # get loss
        ls = CategoricalCrossentropy()(y_true, y_pred)

        # get meatdata
        label = metadata_gt_label(idx, set)
        label_name = metadata_gt_name(idx, set)
        black_pixels = metadata_count_black_pixels(idx, set)
        pixels_count = metadata_pixels_count(idx, set)

        # get visualizer
        image = concat.astype(np.float32) * 255
        histogram = np.zeros(256, dtype=np.float32)
        for pixel in image.reshape(-1):
            histogram[pixel.astype(int)] += 1

        sorted_indices = np.argsort(histogram)[::-1]
        top_50_values = histogram[sorted_indices[:50]]
        top_50_indices = sorted_indices[:50].astype(np.float32)
        top_50_values_str = [str(number) for number in top_50_values.tolist()]
        top_50_indices_str = [str(number) for number in top_50_indices.tolist()]

        # Plot the histogram
        x_values = np.arange(256)
        plt.figure(figsize=(8, 6))
        plt.bar(x_values, histogram, color='gray', alpha=0.7)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Grayscale Histogram')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


if __name__ == "__main__":
    check_custom_integration()
