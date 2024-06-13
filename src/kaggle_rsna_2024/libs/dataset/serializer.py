import pathlib
import pandas as pd
import tensorflow as tf
import shutil

from kaggle_rsna_2024.libs.scan_type import ScanType

class InputDataItemSerializer:
    def __init__(self, default_value):
        self.default_value = default_value

    def serialize(self, input_data_item, label):
        features = {
            f'image_{scan_type}': self._bytes_feature([
                tf.io.serialize_tensor(
                    input_data_item.scans[scan_type][0].get_image_array()
                ).numpy()
            ])
            for scan_type in ScanType
            if len(input_data_item.scans[scan_type]) > 0
        }
        features['label'] = self._bytes_feature([tf.io.serialize_tensor(label).numpy()])
        features['study_id'] = self._ints_feature([input_data_item.info.study_id])
        features.update({
            f"shape_{scan_type}": self._ints_feature(input_data_item.scans[scan_type][0].info.shape)
            for scan_type in ScanType
            if len(input_data_item.scans[scan_type]) > 0
        })

        default_features = {
            f'image_{scan_type}': self._bytes_feature([tf.io.serialize_tensor(self.default_value).numpy()])
            for scan_type in ScanType
            if len(input_data_item.scans[scan_type]) == 0
        }

        features.update(default_features)

        train_example = tf.train.Example(features=tf.train.Features(feature=features))
        return train_example.SerializeToString()

    def _bytes_feature(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def _ints_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    

class TFRecordWriter:
    def __init__(self, directory: pathlib.Path, file_size = 10 ** 8):
        self.directory = directory
        self.writer_file_size = file_size
        self.current_file_size = self.writer_file_size
        self.current_file_index = 0
        self.writer = None

        if self.directory.exists():
            shutil.rmtree(str(self.directory))
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        if self.current_file_size >= self.writer_file_size:
            self.__create_new_writer()

        assert (self.current_file_size < self.writer_file_size)
        assert (self.writer is not None)

        self.writer.write(record)
        self.current_file_size += len(record)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def get_count_files(self):
        return self.current_file_index

    def __create_new_writer(self):
        self.flush()

        self.writer = tf.io.TFRecordWriter(
            str(self.directory / "{:02}.tfrecords".format(self.current_file_index)))
        self.current_file_size = 0
        self.current_file_index += 1