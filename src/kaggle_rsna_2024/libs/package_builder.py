import os
import tempfile
from distutils.dir_util import copy_tree


class PackageBuilder:
    def __init__(self):
        pass

    def build(self, directory):
        current_working_dir = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        
        os.chdir(directory)
        os.system(f"python setup.py sdist bdist_wheel")
        os.chdir(current_working_dir)

        copy_tree(f"{directory}/dist/", temp_dir)

        return temp_dir