import os
from pydicom import dcmread
import numpy as np
import SimpleITK as sitk

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

class Input3dScan:
    def __init__(self, directory: Path, scan_type: ScanType):
        self.directory = directory
        self.scan_type = scan_type
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(self.directory))
        reader.SetFileNames(dicom_names)
        self.image = reader.Execute()
        self.image_array = sitk.GetArrayFromImage(self.image)

    def get_image(self):
        return self.image

    def get_image_array(self):
        return self.image_array
