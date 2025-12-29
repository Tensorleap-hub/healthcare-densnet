from code_loader.plot_functions.visualize import visualize

from leap_binder import *


if model_name == 'DenseNet201':
    labels = CONFIG['LABELS_DenseNet201']
else:
    labels = CONFIG['LABELS']


prediction_type1 = PredictionTypeHandler('classes', labels,channel_dim=-1)





@tensorleap_load_model([prediction_type1])
def load_model():
    H5_MODEL_PATH = "model/cnn.h5"
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, H5_MODEL_PATH)
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(dir_path, model_path))



@tensorleap_integration_test()
def check_custom_integration(idx, preprocess_response: PreprocessResponse):
    print("Starting custom tests")
    model = load_model()


    image_input = input_encoder(idx, preprocess_response)

    gt = gt_encoder(idx, preprocess_response)

    y_pred = model([image_input])

    ls = categorical_crossentropy_loss(gt, y_pred)

    visualizer_pixels_histogram_ = visualizer_pixels_histogram(image_input)
    bar_visualizer_gt = bar_visualizer(gt)
    bar_visualizer_pred = bar_visualizer(y_pred)

    visualize(visualizer_pixels_histogram_)
    visualize(bar_visualizer_gt)
    visualize(bar_visualizer_pred)

    label = metadata_gt_label(idx, preprocess_response)
    black_pixels = metadata_count_black_pixels(idx, preprocess_response)
    pixels_count = metadata_pixels_count(idx, preprocess_response)

    label_name = metadata_gt_name(idx, preprocess_response)



if __name__ == '__main__':
    responses = preprocess_func()
    train = responses[0]
    check_custom_integration(0, train)