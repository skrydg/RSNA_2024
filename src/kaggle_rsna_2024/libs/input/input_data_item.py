from pathlib import Path
import os

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScan

class InputDataItem:
    def __init__(self, directory: Path, series_info: dict):
        self.directory = directory
        self.scans = {}
        for series_id, scan_type in series_info.items():
            series_directory = self.directory / str(series_id)
            self.scans[scan_type] = Input3dScan(series_directory, series_id, scan_type)
