import polars as pl

from collections import defaultdict

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.scan_type import ScanType, string_to_scan_type

from kaggle_rsna_2024.libs.input.input_data_item import InputDataItem

class DataLoader:
    def __init__(self, env: Env, image_shape):
        self.env = env
        self.image_shape = image_shape
        self.train_series_descriptions = self._load_series_descriptions(env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / "train_series_descriptions.csv")
        self.test_series_descriptions = self._load_series_descriptions(env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification"/ "test_series_descriptions.csv")

    def get_train_study_ids(self):
        return self.train_series_descriptions.keys()

    def get_test_study_ids(self):
        return self.test_series_descriptions.keys()

    def stream_train_data(self):
        for study_id, study_desc in self.train_series_descriptions.items():
            yield InputDataItem(
                self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / "train_images" / str(study_id), 
                study_id, 
                study_desc,
                self.image_shape
            )

    def stream_test_data(self):
        for study_id, study_desc in self.test_series_descriptions.items():
            yield InputDataItem(
                self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification"/ "test_images" / str(study_id),
                study_id, 
                study_desc,
                self.image_shape
            )

    def get_train_item(self, study_id):
        study_desc = self.train_series_descriptions[study_id]
        return InputDataItem(
            self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / "train_images" / str(study_id), 
            study_id,
            study_desc,
            self.image_shape
        )

    def get_test_item(self, study_id):
        study_desc = self.test_series_descriptions[study_id]
        return InputDataItem(
            self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" /"test_images" / str(study_id), 
            study_id, 
            study_desc,
            self.image_shape
        )

    def _load_series_descriptions(self, filepath):
        csv_series_descriptions = pl.read_csv(filepath)
        series_descriptions = defaultdict(lambda: defaultdict())
        for (study_id, series_id, series_description) in csv_series_descriptions.rows():
            series_descriptions[study_id][series_id] = string_to_scan_type(series_description)
        return series_descriptions