import polars as pl

from kaggle_rsna_2024.libs.env import Env
from kaggle_rsna_2024.libs.input import DataLoaderConfiguration, DatasetType

class LabelLoader:
    COLUMNS = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']
    LEVELS = ['Normal/Mild', 'Moderate', 'Severe']

    def __init__(self, data_loader_configuration: DataLoaderConfiguration):
        self.DUMMY_COLUMNS = [f"{column}_{level}" for column in COLUMNS for level in LEVELS]
        self.data_loader_configuration = data_loader_configuration

    def load(self, study_ids):
        train = pl.read_csv(self.data_loader_configuration.get_labels_path(DatasetType.Train))
        train = train.filter(pl.col("study_id").is_in(study_ids))

        train = train.fill_null("Normal/Mild")
        train = train.to_dummies(self.COLUMNS)
        
        for column in self.DUMMY_COLUMNS:
            if column not in train.columns:
                train = train.with_columns(pl.lit(0).alias(column))
        train = train[["study_id"] + self.DUMMY_COLUMNS]
        
        rows = []
        for study_id in study_ids:
            rows.append(train.filter(pl.col("study_id") == study_id))
        
        columns = train.columns
        columns.remove("study_id")
        return pl.concat(rows, how='vertical')[self.COLUMNS].to_numpy()
