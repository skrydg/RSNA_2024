import os
from pydicom import dcmread
import numpy as np
import SimpleITK as sitk

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

def reshape_image(image, new_size):
    shape_size = len(image.GetSize())
    new_spacing = [
        image.GetSize()[i] * image.GetSpacing()[i] / new_size[i]
        for i in range(shape_size)
    ]

    image = sitk.Resample(
        image, new_size, sitk.Transform(),
        sitk.sitkLinear, image.GetOrigin(), new_spacing,
        image.GetDirection(), 0.0, image.GetPixelID()
    )

    return image

class Input3dScanMetaInfo:
    def __init__(self,
                 directory: Path,
                 study_id: int,
                 series_id: int,
                 scan_type: ScanType,
                 shape: tuple):
        self.directory = directory
        self.study_id = study_id
        self.series_id = series_id
        self.scan_type = scan_type
        self.shape = shape


class Input3dScan:
    def __init__(self, info: Input3dScanMetaInfo, image_shapes: list):
        self.info = info
        self.image_shapes = image_shapes
        self.images = {}

        for image_shape in self.image_shapes:
            scans = []
            dicom_names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(self.info.directory))
            reader = sitk.ImageSeriesReader()
            for dicom_name in dicom_names:
                reader.SetFileNames([dicom_name])
                image = reader.Execute()
                image_shape_2d = list(image_shape[:2]) + [1]
                scans.append(sitk.GetArrayFromImage(reshape_image(image, image_shape_2d))[0, :, :])

            scans = np.array(scans)
            image = sitk.GetImageFromArray(scans)
            image = reshape_image(image, image_shape)
            self.images[str(image_shape)] = image

    def get_image(self, shape):
        return self.images[str(shape)]

    def get_image_array(self, shape):
        return self._convert_to_8bit(sitk.GetArrayFromImage(self.get_image(shape)))
    
    def _convert_to_8bit(self, x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x) 
        return (x * 255).astype("uint8")
