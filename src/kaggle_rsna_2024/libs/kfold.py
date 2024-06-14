import numpy as np

class TFDatasetKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_batch_getter(self, i):
        def get_batch(index, r):
            return index % self.n_splits == i
        return get_batch
    
    def get_out_batch_getter(self, i):
        def get_out_batch(index, r):
            return index % self.n_splits != i
        return get_out_batch
    
    def split(self, dataset):
        dataset_size = dataset.reduce(0, lambda x, r: x + 1).numpy()
        batch_size = dataset_size // self.n_splits
        batches = [batch_size] * (self.n_splits - 1)
        batches.append(dataset_size - batch_size * (self.n_splits - 1))
        batch_size = dataset_size // self.n_splits
        for i in range(self.n_splits):
            train_dataset = dataset.enumerate().filter(self.get_out_batch_getter(i)).map(lambda index, r: r)
            test_dataset = dataset.enumerate().filter(self.get_batch_getter(i)).map(lambda index, r: r)
            yield train_dataset, test_dataset