import json
import os
import shutil
import tempfile

import json

from pathlib import Path
from kaggle.rest import ApiException
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleApiClient:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.username = self.api.get_config_value(self.api.CONFIG_NAME_USER)

    def upload_dataset(self, title, directory) -> None:
        if self._is_dataset_exists(title):
            self._upload_dataset_version(title, directory)
        else:
            self._upload_dataset(title, directory)
 

    def _is_dataset_exists(self, title) -> bool:
        dataset_list = self.api.dataset_list(mine=True)
        return title in [dataset.title for dataset in dataset_list]
    
    def _upload_dataset(self, title, directory) -> None:
        try:
            self._create_meta_file(directory, title)
            self.api.dataset_create_new_cli(directory, dir_mode='zip')
        except ApiException as e:
            print(f"Exception when calling KaggleApi->datasets_create_new: {e}")

    def _upload_dataset_version(self, title, directory) -> None:
        try:
            self._create_meta_file(directory, title)
            self.api.dataset_create_version(directory, version_notes="", dir_mode="zip")
        except ApiException as e:
            print(f"Exception when calling KaggleApi->datasets_create_new: {e}")
            
    def _create_meta_file(self, directory, title):
        self.api.dataset_initialize(directory)
        meta_data_path = self.api.get_dataset_metadata_file(directory)
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)
        meta_data["title"] = title
        meta_data["id"] = f"{self.username}/{title}"
        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)