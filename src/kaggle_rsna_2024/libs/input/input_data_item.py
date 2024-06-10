from pathlib import Path
import os

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScan

class InputDataItem:
    def __init__(self, directory: Path, study_id: int, series_info: dict, image_shape: tuple):
        self.directory = directory
        self.study_id = study_id
        self.image_shape = image_shape

        self.scans = {}
        for series_id, scan_type in series_info.items():
            series_directory = self.directory / str(series_id)
            self.scans[scan_type] = Input3dScan(series_directory, study_id, series_id, scan_type, image_shape)
