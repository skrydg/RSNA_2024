import glob
import tensorflow as tf
import numpy as np

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

class InputDataItemDeserializer:
    def __init__(self, default_value):
        self.default_value = default_value
    
    def deserialize(self, record_bytes):
        features = {
            f"image_{scan_type}": tf.io.FixedLenFeature([], tf.string)
            for scan_type in ScanType
        }
        features["study_id"] = tf.io.FixedLenFeature([], tf.int64)
        features["label"] = tf.io.FixedLenFeature([], tf.string)

        example = tf.io.parse_single_example(
            record_bytes,
            features = features
        )

        return ([
                tf.io.parse_tensor(example[f'image_{scan_type}'], out_type=tf.uint8)
                for scan_type in ScanType
            ],
            tf.io.parse_tensor(example["label"], out_type=tf.int32),
            example["study_id"]
        )
    
class TFRecordReader:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def read(self):
        assert (self.directory.exists())
        return tf.data.TFRecordDataset(self.__get_files())

    def __get_files(self):
        return glob.glob(f'{self.directory}/*/*.tfrecords')