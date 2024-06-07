import polars as pl
import os

from collections import defaultdict
from pydicom import dcmread

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.scan_type import string_to_scan_type

class MetaDataLoader:
    def __init__(self, env: Env, limit=1000000000):
        self.env = env
        self.limit = limit
        self.train_series_descriptions_file = env.input_directory / "train_series_descriptions.csv"
        self.test_series_descriptions_file = env.input_directory / "test_series_descriptions.csv"
        self.train_images = env.input_directory / "train_images"
        self.test_images = env.input_directory / "test_images"
        
    def get_train_meta_data(self):
        return self.load_meta_data(self.train_series_descriptions_file, self.train_images)
    
    def get_test_meta_data(self):
        return self.load_meta_data(self.test_series_descriptions_file, self.test_images)
    
    def load_meta_data(self, series_descriptions_file, images_directory):
        series_descriptions = self._load_series_descriptions(series_descriptions_file)
        meta_data = []
        
        count_items = 0
        for study_id, study_desc in series_descriptions.items():
            count_items += 1
            if (count_items >= self.limit):
                break
            for series_id, scan_type in study_desc.items():
                series_directory =  images_directory / str(study_id) / str(series_id)
                for filename in os.listdir(series_directory):
                    ds = dcmread(series_directory / filename, stop_before_pixels=True)
                    current_meta_data = {}
                    for i in ds:
                        current_meta_data[i.description()] =  i.value
                    meta_data.append(current_meta_data)
        return pl.DataFrame(meta_data)

    def _load_series_descriptions(self, filepath):
        csv_series_descriptions = pl.read_csv(filepath)
        series_descriptions = defaultdict(lambda: defaultdict())
        for (study_id, series_id, series_description) in csv_series_descriptions.rows():
            series_descriptions[study_id][series_id] = string_to_scan_type(series_description)
        return series_descriptions