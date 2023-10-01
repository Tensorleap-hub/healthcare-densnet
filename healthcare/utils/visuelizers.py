import numpy as np
from typing import List, Union

from healthcare.config import CONFIG


def pixels_histogram(image: np.ndarray) -> Union[np.ndarray, List]:
    image = image.astype(np.float32) * 255
    histogram = np.zeros(256, dtype=np.float32)
    for pixel in image.reshape(-1):
        histogram[pixel.astype(int)] += 1
    sorted_indices = np.argsort(histogram)[::-1]
    top_K_values = histogram[sorted_indices[:CONFIG['K']]]
    top_K_indices = sorted_indices[:CONFIG['K']].astype(np.float32)
    top_K_values_str = [str(number) for number in top_K_values.tolist()]

    return top_K_indices, top_K_values_str
