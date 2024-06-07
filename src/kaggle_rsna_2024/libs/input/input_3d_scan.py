import os
from pydicom import dcmread
import numpy as np

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

class Input3dScan:
    def __init__(self, directory: Path, scan_type: ScanType):
        self.directory = directory
        self.scan_type = scan_type
        
        indexes = sorted([int(Path(filename).stem) for filename in os.listdir(self.directory)])
        self.series = [dcmread(directory / f"{index}.dcm").pixel_array for index in indexes]
#        self.series = np.array(self.series)