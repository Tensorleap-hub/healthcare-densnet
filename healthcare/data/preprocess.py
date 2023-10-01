import os
from typing import List, Tuple, Any


def load_images(local_filepath) -> Tuple[List[Any], List[List[int]]]:
    file_path_list = []
    label_list = []

    for value in ["bacteria", "NORMAL", "virus"]:
        if value == "NORMAL":
            label = [1, 0, 0]
        elif value == "bacteria":
            label = [0, 1, 0]
        elif value == "virus":
            label = [0, 0, 1]
        else:
            raise ValueError("Invalid value: " + value + ". Value should be bacteria /NORMAL /virus")

        for filename in os.listdir(local_filepath.joinpath(value)):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = (local_filepath.joinpath(value)).joinpath(filename)
                file_path_list.append(file_path)
                label_list.append(label)
    return file_path_list, label_list
