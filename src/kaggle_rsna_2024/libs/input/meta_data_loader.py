import pandas as pd
import os
import pydicom

from collections import defaultdict
from pydicom import dcmread


from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.scan_type import string_to_scan_type
from kaggle_rsna_2024.libs.input import DataLoaderConfiguration, DatasetType

class MetaDataLoader:
    def __init__(self, data_loader_configuration: DataLoaderConfiguration, limit=1000000000):
        self.data_loader_configuration = data_loader_configuration
        self.limit = limit
        

    def get_train_meta_data(self):
        return self.get_meta_data(DatasetType.Train)

    def get_test_meta_data(self):
        return self.get_meta_data(DatasetType.Test)

    def get_meta_data(self, dataset_type):
        images_directory = self.data_loader_configuration.get_image_path(dataset_type)
        series_descriptions = self._load_series_descriptions(dataset_type)
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

                        if type(i.value) is pydicom.multival.MultiValue:
                            current_meta_data[i.description()] = list(i.value)
                        else:
                            current_meta_data[i.description()] = i.value
                        #print(i.description(), list(i.value), type(i.value), i.value.__dict__)
                    meta_data.append(current_meta_data)
        return pd.DataFrame(meta_data)

    def _load_series_descriptions(self, dataset_type):
        filepath = self.data_loader_configuration.get_series_description_path(dataset_type)

        csv_series_descriptions = pl.read_csv(filepath)
        series_descriptions = defaultdict(lambda: defaultdict())
        for (study_id, series_id, series_description) in csv_series_descriptions.rows():
            series_descriptions[study_id][series_id] = string_to_scan_type(series_description)
        return series_descriptions