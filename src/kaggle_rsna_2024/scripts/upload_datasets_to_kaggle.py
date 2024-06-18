import shutil
from kaggle_rsna_2024.libs.package_builder import PackageBuilder
from kaggle_rsna_2024.libs.kaggle.api import KaggleApiClient

directory = PackageBuilder().build(".")
api = KaggleApiClient()
api.upload_dataset("RSNA-2024-code", directory)
shutil.rmtree(directory)