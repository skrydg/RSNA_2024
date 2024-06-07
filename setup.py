from setuptools import setup, find_packages

setup(
  name='kaggle_rsna_2024',
  version='0.1',
  packages=find_packages(where="src"),
  package_dir={"": "src"},
)