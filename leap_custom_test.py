import os
import tensorflow as tf
from keras.losses import CategoricalCrossentropy

from leap_binder import *
from code_loader.helpers import visualize


def check_custom_integration():
    plot_vis = True
    check_generic = True

    if check_generic:
        leap_binder.check()

    preprocess = preprocess_func()
    train = preprocess[0]
    val = preprocess[1]
    test = preprocess[2]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = f"model/{model_name}.h5"
    model = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    set = train
    for idx in range(5):

        # get input and gt
        if model_name == 'cnn':
            image_input = input_encoder_cnn(idx, set)

        else:
            image_input = input_encoder(idx, set)

        concat = np.expand_dims(image_input, axis=0)
        gt = gt_encoder(idx, set)
        y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

        y_pred = model([concat])

        # get loss
        ls = CategoricalCrossentropy()(y_true, y_pred)

        # get visualizer
        visualizer_pixels_histogram_ = visualizer_pixels_histogram(concat)
        bar_visualizer_gt = bar_visualizer(np.expand_dims(gt, 0))
        bar_visualizer_pred = bar_visualizer(y_pred.numpy())

        if plot_vis:
            visualize(visualizer_pixels_histogram_)
            visualize(bar_visualizer_gt)
            visualize(bar_visualizer_pred)

        # get metadata
        label = metadata_gt_label(idx, set)
        black_pixels = metadata_count_black_pixels(idx, set)
        pixels_count = metadata_pixels_count(idx, set)
        if model_name == 'DenseNet201':
            label_name = metadata_gt_name_DenseNet201(idx, set)
        else:
            label_name = metadata_gt_name(idx, set)



        print(f"idx {idx} successfully finish")


if __name__ == "__main__":
    check_custom_integration()
