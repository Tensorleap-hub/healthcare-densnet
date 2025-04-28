import os
from typing import List, Tuple, Any
from pathlib import Path

from healthcare.config import CONFIG
from healthcare.utils.gcs_utils import _connect_to_gcs_and_return_bucket

model_name = CONFIG['model'][2]

def load_images(set) -> Tuple[List[Any], List[List[int]]]:
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dataset_path = Path(CONFIG['cloud_file_path'])
    image_list = []
    labels_list = []

    if model_name == 'DenseNet201':
        for value in ["NORMAL", "bacteria", "virus"]:
            if value == "NORMAL":
                label = [1, 0, 0]
            elif value == "bacteria":
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            image_list_per_value = [obj.name for obj in bucket.list_blobs(prefix=str(dataset_path / set / value))]
            image_list_per_value = image_list_per_value[1: len(image_list_per_value)]
            labels_list_per_value = [label for _ in range(len(image_list_per_value))]

            image_list = image_list + image_list_per_value
            labels_list = labels_list + labels_list_per_value

    else:
        for value in ["NORMAL", "Pneumonia"]:
            if value == "NORMAL":
                label = [1, 0]
                image_list_per_value = [obj.name for obj in bucket.list_blobs(prefix=str(dataset_path / set / value))]

            else:
                label = [0, 1]
                image_list_per_value = [obj.name for obj in bucket.list_blobs(prefix=str(dataset_path / set / "bacteria"))]
                image_list_per_value_1 = image_list_per_value[1: len(image_list_per_value)]
                image_list_per_value = [obj.name for obj in bucket.list_blobs(prefix=str(dataset_path / set / "virus"))] \
                                       + image_list_per_value_1

            image_list_per_value = image_list_per_value[1: len(image_list_per_value)]
            labels_list_per_value = [label for _ in range(len(image_list_per_value))]

            image_list = image_list + image_list_per_value
            labels_list = labels_list + labels_list_per_value

    return image_list, labels_list
