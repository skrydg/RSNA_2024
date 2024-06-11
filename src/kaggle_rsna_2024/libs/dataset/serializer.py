import pathlib
import pandas as pd
import tensorflow as tf
import shutil


from kaggle_rsna_2024.libs.scan_type import ScanType

class InputDataItemSerializer:
    def serialize(self, input_data_item):
        image = input_data_item.scans[ScanType.sagittal_t2_stir][0].get_image_array()
        train_example = tf.train.Example(features=tf.train.Features(                          
            feature={
                'image': self._bytes_feature([tf.io.serialize_tensor(image).numpy()]),         
            }))
        return train_example.SerializeToString()

    def _bytes_feature(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


class TFRecordWriter:
    def __init__(self, directory: pathlib.Path, batch_size = 100):
        self.directory = directory
        self.writer_batch_size = batch_size
        self.current_batch_size = self.writer_batch_size
        self.current_file_index = 0
        self.writer = None

    def write(self, record):
        if self.current_batch_size >= self.writer_batch_size:
            self.__create_new_writer()

        assert (self.current_batch_size < self.writer_batch_size)
        assert (self.writer is not None)

        self.writer.write(record)
        self.current_batch_size += len(record)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def get_count_files(self):
        return self.current_file_index

    def __create_new_writer(self):
        self.flush()

        self.writer = tf.io.TFRecordWriter(
            str(self.directory / "{:02}.tfrecords".format(self.current_file_index)))
        self.current_batch_size = 0
        self.current_file_index += 1