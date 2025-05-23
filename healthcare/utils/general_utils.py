import cv2
import numpy as np

from healthcare.config import CONFIG


def load_image(file_path) -> np.ndarray:
    img = cv2.imread(file_path)
    img = cv2.resize(img, (CONFIG['img_dims'], CONFIG['img_dims']))
    return img


def flat_image(img: np.ndarray) -> np.ndarray:
    grayscale_image = np.dot(img[..., :3], CONFIG['scaling_factors'])
    flattened_image = grayscale_image.flatten().astype(int)
    return flattened_image
