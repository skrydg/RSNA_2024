from pathlib import Path
import os

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScan, Input3dScanMetaInfo
from collections import defaultdict
from typing import Dict, Tuple

class InputDataItemMetaInfo:
    def __init__(self, directory: Path, study_id: int, scan_meta_info: Dict[int, Input3dScanMetaInfo]):
        self.directory = directory
        self.study_id = study_id
        self.scan_meta_info = scan_meta_info


class InputDataItem:
    def __init__(self, info: InputDataItemMetaInfo, image_shapes: list[Tuple[int]]):
        self.info = info
        self.image_shapes = image_shapes

        self.scans = defaultdict(list)
        for series_id in self.info.scan_meta_info.keys():
            self.scans[self.info.scan_meta_info[series_id].scan_type].append(
                Input3dScan(self.info.scan_meta_info[series_id], image_shapes)
            )
