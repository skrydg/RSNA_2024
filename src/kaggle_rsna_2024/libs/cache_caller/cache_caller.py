import pickle

from pathlib import Path

class CacheCaller:
    def __init__(self, impl, serializer, directory):
        self.impl = impl
        self.serializer = serializer
        self.directory = Path(directory)
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)

    def _call(self, name):
        filename = self.directory / f"{self.impl.__class__.__name__}_{str(name)}"
        if filename.exists():
            print("Load data from cache", flush=True)
            return self.serializer.deserialize(filename)
            
        print("No data found in cache", flush=True)
        res = getattr(self.impl, name)()
        self.serializer.serialize(res, filename)
        return res

    def __getattr__(self, name):
        return (lambda: self._call(name))
