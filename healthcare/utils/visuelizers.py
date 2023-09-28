import numpy as np
from typing import List, Union


def pixels_histogram(image: np.ndarray) -> Union[np.ndarray, List]:
    image = image.astype(np.float32) * 255
    histogram = np.zeros(256, dtype=np.float32)
    for pixel in image.reshape(-1):
        histogram[pixel.astype(int)] += 1
    sorted_indices = np.argsort(histogram)[::-1]
    top_50_values = histogram[sorted_indices[:30]]
    top_50_indices = sorted_indices[:30].astype(np.float32)
    top_50_values_str = [str(number) for number in top_50_values.tolist()]

    return top_50_indices, top_50_values_str
