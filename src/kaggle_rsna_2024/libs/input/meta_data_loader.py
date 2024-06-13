import pandas as pd
import os
import pydicom

from collections import defaultdict
from pydicom import dcmread

from multiprocessing import Pool
from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.scan_type import string_to_scan_type
from kaggle_rsna_2024.libs.input import DataLoaderConfiguration, DatasetType

class MetaDataLoader:
    def __init__(self, data_loader_configuration: DataLoaderConfiguration, limit=1000000000):
        self.data_loader_configuration = data_loader_configuration
        self.limit = limit
        self.pool_size = 8

    def get_train_meta_data(self):
        return self.get_meta_data(DatasetType.Train)

    def get_test_meta_data(self):
        return self.get_meta_data(DatasetType.Test)

    def get_meta_data_batch(self, args):
        batch_index, dataset_type = args
        images_directory = self.data_loader_configuration.get_image_path(dataset_type)
        series_descriptions = self._load_series_descriptions(dataset_type)
        
        batch_study_ids = list(series_descriptions.keys())[batch_index::self.pool_size]
        count_items = 0
        meta_data = []
        for study_id in batch_study_ids:
            study_desc = series_descriptions[study_id]
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
                    meta_data.append(current_meta_data)
        return meta_data
    
    def get_meta_data(self, dataset_type):
        with Pool(self.pool_size) as p:
            meta_data = p.map(
                self.get_meta_data_batch,
                zip(list(range(self.pool_size)), [dataset_type for i in range(self.pool_size)])
            )
        
        meta_data = sum(meta_data, [])
        return pd.DataFrame(meta_data)

    def _load_series_descriptions(self, dataset_type):
        filepath = self.data_loader_configuration.get_series_description_path(dataset_type)

        csv_series_descriptions = pd.read_csv(filepath)
        series_descriptions = defaultdict(lambda: defaultdict())
        for index, row in csv_series_descriptions.iterrows():
            series_descriptions[row["study_id"]][row["series_id"]] = string_to_scan_type(row["series_description"])
        return series_descriptions