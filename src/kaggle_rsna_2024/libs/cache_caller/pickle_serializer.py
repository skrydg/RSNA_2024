import pickle

class PickleSerializer:
    def serialize(self, data, filename):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def deserialize(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)