from kaggle_rsna_2024.libs.env import Env

from pathlib import Path
from enum import Enum

class DatasetType(Enum):
    Train = 0,
    Test = 1

class DataLoaderConfiguration:
    def __init__(self, env: Env, dataset_type_to_name = None):
        self.env = env
        if dataset_type_to_name is None:
            dataset_type_to_name = {
                DatasetType.Train: "train",
                DatasetType.Test: "test"
            }
        self.dataset_type_to_name = dataset_type_to_name

    def get_image_path(self, dataset_type: DatasetType) -> Path:
        return self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / f"{self.dataset_type_to_name[dataset_type]}_images"

    def get_series_description_path(self, dataset_type: DatasetType) -> Path:
        return self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / f"{self.dataset_type_to_name[dataset_type]}_series_descriptions.csv"

    def get_labels_path(self, dataset_type: DatasetType) -> Path:
        return self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / f"{self.dataset_type_to_name[dataset_type]}.csv"