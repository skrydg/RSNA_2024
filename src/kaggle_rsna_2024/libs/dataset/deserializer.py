import glob
import tensorflow as tf
import numpy as np

from pathlib import Path

from kaggle_rsna_2024.libs.scan_type import ScanType

class InputDataDeserializer:
    def __init__(self, default_value):
        self.default_value = default_value
    
    def deserialize(self, record_bytes):
        default_value = tf.io.serialize_tensor(np.zeros(shape=[20, 256, 256], dtype=np.uint8))
        
        example = tf.io.parse_single_example(
            record_bytes,
            features = {
                f"image_{scan_type}": tf.io.FixedLenFeature([], tf.string)
                for scan_type in ScanType
            }
        )
        return (tf.io.parse_tensor(
            example['image'], out_type=tf.uint8
        ),
        tf.io.parse_tensor(
            example['image2'], out_type=tf.uint8
        ))
        

class TFRecordReader:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def read(self):
        assert (self.directory.exists())
        return tf.data.TFRecordDataset(self.__get_files())

    def __get_files(self):
        return glob.glob(f'{self.directory}/*.tfrecords')