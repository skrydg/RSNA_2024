import os
from pydicom import dcmread
import numpy as np
import SimpleITK as sitk

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

class Input3dScan:
    def __init__(self, directory: Path, series_id: int, scan_type: ScanType):
        self.directory = directory
        self.series_id = series_id
        self.scan_type = scan_type
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(self.directory))
        reader.SetFileNames(dicom_names)

        self.image = reader.Execute()

    def get_image(self):
        return self.image

    def resample(self, new_size):
        new_spacing = [
            self.image.GetSize()[i] * self.image.GetSpacing()[i] / new_size[i]
            for i in range(3)
        ]

        self.image = sitk.Resample(
            self.image, new_size, sitk.Transform(),
            sitk.sitkLinear, self.image.GetOrigin(), new_spacing,
            self.image.GetDirection(), 0.0, self.image.GetPixelID()
        )
        return self

    def get_image_array(self):
        return sitk.GetArrayFromImage(self.image)