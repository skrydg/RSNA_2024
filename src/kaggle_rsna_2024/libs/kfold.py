import numpy as np

class TFDatasetKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, dataset):
        dataset_size = dataset.reduce(0, lambda x, r: x + 1).numpy()
        batch_size = dataset_size // self.n_splits
        batches = [batch_size] * (self.n_splits - 1)
        batches.append(dataset_size - batch_size * (self.n_splits - 1))
        batch_size = dataset_size // self.n_splits
        for i in range(self.n_splits):
            first_train_interval = int(np.sum(batches[0:i]))
            test_interval = int(batches[i])
            second_train_interval = int(np.sum(batches[i + 1:]))
            train_dataset = dataset.take(first_train_interval).skip(test_interval).take(second_train_interval)
            test_dataset = dataset.skip(first_train_interval).take(test_interval)
            yield train_dataset, test_dataset