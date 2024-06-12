import polars as pl

from kaggle_rsna_2024.libs.env import Env

class LabelLoader:
    def __init__(self, env: Env):
        self.env = env
        self.columns = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']
        self.levels = ['Normal/Mild', 'Moderate', 'Severe']
        self.dummy_columns = [f"{column}_{level}" for column in self.columns for level in self.levels]

    def load(self, study_ids):
        train = pl.read_csv(self.env.input_directory / "rsna-2024-lumbar-spine-degenerative-classification" / "train.csv")
        train = train.filter(pl.col("study_id").is_in(study_ids))

        train = train.fill_null("Normal/Mild")
        train = train.to_dummies(self.columns)
        
        for column in self.dummy_columns:
            if column not in train.columns:
                train = train.with_columns(pl.lit(0).alias(column))
        train = train[["study_id"] + self.dummy_columns]
        
        rows = []
        for study_id in study_ids:
            rows.append(train.filter(pl.col("study_id") == study_id))
        
        columns = train.columns
        columns.remove("study_id")
        return pl.concat(rows, how='vertical')[columns].to_numpy()
