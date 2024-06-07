import pickle

from pathlib import Path

class CachedCaller:
    def __init__(self, impl, directory):
        self.impl = impl
        self.directory = Path(directory)
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        
    def __getattr__(self, name):
        filename = self.directory / f"{self.impl.__class__.__name__}_{str(name)}.pkl"
        if filename.exists():
            print("Load data from cache", flush=True)
            with open(filename, 'rb') as file:
                return pickle.load(file)
            
        print("No data found in cache", flush=True)
        res = getattr(self.impl, name)
        with open(filename, 'wb') as file:
            pickle.dump(res, file)
        return res
        