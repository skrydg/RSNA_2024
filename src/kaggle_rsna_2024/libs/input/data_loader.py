import pandas as pd

from collections import defaultdict

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.scan_type import ScanType, string_to_scan_type

from kaggle_rsna_2024.libs.input.data_loader_configuration import DatasetType
from kaggle_rsna_2024.libs.input.data_loader_configuration import DataLoaderConfiguration
from kaggle_rsna_2024.libs.input.meta_data_loader import MetaDataLoader
from kaggle_rsna_2024.libs.input.input_data_item import InputDataItem, InputDataItemMetaInfo
from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScanMetaInfo, Input3dScanMetaInfo
 
class DataLoader:
    def __init__(self, data_loader_configuration: DataLoaderConfiguration, meta_data, image_shapes):
        self.data_loader_configuration = data_loader_configuration
        self.meta_data = meta_data
        self.image_shapes = image_shapes
        self.series_descriptions = {
             dataset_type: self._load_series_descriptions(dataset_type)
             for dataset_type in DatasetType
        }
        
    def get_study_ids(self, dataset_type):
        return list(self.series_descriptions[dataset_type].keys())

    def get_item(self, dataset_type, study_id):
        meta_data = self.get_meta_data(dataset_type, study_id)
        return InputDataItem(meta_data, self.image_shapes)

    def _load_series_descriptions(self, dataset_type):
        filepath = self.data_loader_configuration.get_series_description_path(dataset_type)

        csv_series_descriptions = pd.read_csv(filepath)
        series_descriptions = defaultdict(lambda: defaultdict())
        for index, row in csv_series_descriptions.iterrows():
            series_descriptions[row["study_id"]][row["series_id"]] = string_to_scan_type(row["series_description"])
        return series_descriptions
    
    def get_meta_data(self, dataset_type, study_id):
        directory = self.data_loader_configuration.get_image_path(dataset_type) / str(study_id)
        study_desc = self.series_descriptions[dataset_type][study_id]
        scan_meta_info = {}
        for series_id, scan_type in study_desc.items():
            scan_directory = directory / str(series_id)
            mask = (self.meta_data["Series Instance UID"] == f"{study_id}.{series_id}")
            rows = self.meta_data[mask]["Rows"].to_numpy()[0]
            columns = self.meta_data[mask]["Columns"].to_numpy()[0]
            height = self.meta_data[mask]["Instance Number"].shape[0]
            scan_meta_info[series_id] = Input3dScanMetaInfo(scan_directory, study_id, series_id, scan_type, (rows, columns, height))

        return InputDataItemMetaInfo(directory, study_id, scan_meta_info)
