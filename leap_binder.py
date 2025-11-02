from code_loader.default_metrics import categorical_crossentropy
from numpy import ndarray
from healthcare.config import CONFIG
from healthcare.data.preprocess import load_images, model_name
from healthcare.utils.gcs_utils import _download
from healthcare.utils.general_utils import load_image, flat_image
from healthcare.utils.visuelizers import pixels_histogram
import cv2


from code_loader.inner_leap_binder.leapbinder_decorators import *

if model_name == 'DenseNet201':
    labels = CONFIG['LABELS_DenseNet201']
else:
    labels = CONFIG['LABELS']

@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    train_X, train_Y = load_images('train')
    val_X, val_Y = load_images('val')
    test_X, test_Y = load_images('test')

    train_length = len(train_X)
    val_length = len(val_X)
    test_length = len(test_X)

    train = PreprocessResponse(length=train_length, data={'images': train_X, 'labels': train_Y})
    val = PreprocessResponse(length=val_length, data={'images': val_X, 'labels': val_Y})
    test = PreprocessResponse(length=test_length, data={'images': test_X, 'labels': test_Y})
    response = [train, val, test]
    return response

def input_encoder_cnn(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    cloud_path = preprocess.data['images'][idx]
    fpath = _download(str(cloud_path))
    img = load_image(fpath)
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_image_3d = grayscale_image[:, :, np.newaxis]
    img = grayscale_image_3d / 255
    return img.astype('float32')

@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    if model_name == 'cnn':
        return input_encoder_cnn(idx, preprocess)

    cloud_path = preprocess.data['images'][idx]
    fpath = _download(str(cloud_path))
    img = load_image(fpath)
    img = img / 255
    return img.astype('float32')



@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.array(preprocess.data['labels'][idx]).astype(np.float32)


@tensorleap_metadata("gt_label")
def metadata_gt_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = np.array(gt_encoder(idx, preprocess))
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int


def metadata_gt_name_DenseNet201(idx: int, preprocess: PreprocessResponse) -> str:
    digit = gt_encoder(idx, preprocess)
    if digit[0] == 1:
        return 'NORMAL'
    elif digit[1] == 1:
        return 'bacteria'
    elif digit[2] == 1:
        return 'virus'
    else:
        raise ValueError("Invalid label value")

@tensorleap_metadata("gt_name")
def metadata_gt_name(idx: int, preprocess: PreprocessResponse) -> str:
    if model_name == 'DenseNet201':
        return metadata_gt_name_DenseNet201(idx, preprocess)

    digit = gt_encoder(idx, preprocess)
    if digit[0] == 1:
        return 'NORMAL'
    elif digit[1] == 1:
        return 'Pneumonia'
    else:
        raise ValueError("Invalid label value")


@tensorleap_metadata("number_of_black_pixels")
def metadata_count_black_pixels(idx: int, preprocess: PreprocessResponse) -> ndarray:
    cloud_path = preprocess.data['images'][idx]
    fpath = _download(str(cloud_path))
    img = load_image(fpath)
    flattened_image = flat_image(img)
    black_pixel_count = np.sum(flattened_image == [0], axis=-1)

    return black_pixel_count


@tensorleap_metadata("pixels_count")
def metadata_pixels_count(idx: int, preprocess: PreprocessResponse) -> Dict:
    cloud_path = preprocess.data['images'][idx]
    fpath = _download(str(cloud_path))
    img = load_image(fpath)
    flattened_image = flat_image(img)
    pixel_counts = {str(key): 0 for key in range(256)}

    for pixel_value in flattened_image:
        pixel_value_int = int(pixel_value)
        pixel_counts[str(pixel_value_int)] += 1

    return pixel_counts


@tensorleap_custom_visualizer('pixels_histogram', LeapHorizontalBar.type)
def visualizer_pixels_histogram(image: np.ndarray) -> LeapHorizontalBar:
    image = np.squeeze(image)
    top_50_indices, top_50_values_str = pixels_histogram(image)
    return LeapHorizontalBar(np.array(top_50_indices), top_50_values_str)


@tensorleap_custom_visualizer('bar_names_visualizer', LeapHorizontalBar.type)
def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    data = data.astype(np.float32)
    data = np.squeeze(data)
    return LeapHorizontalBar(data, labels)


@tensorleap_custom_loss('categorical_crossentropy_loss')
def categorical_crossentropy_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)








