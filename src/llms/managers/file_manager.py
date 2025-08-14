import ast

import pandas as pd
import os

class FileManager:
    def __init__(self, file_path: str, metrics_file_path: str = None):
        self.pd_file_path = file_path
        self.metrics_file_path = metrics_file_path
        self.df = None
        self.df_exists = os.path.exists(self.pd_file_path)
        self.load_data()
        self.null_index = 0
        self.last_column = None

        self.main_df_columns = [
            "prompt",
            "prompt_index",
        ]
        self.main_df_columns.extend([f"response_{i + 1}" for i in range(3)])
        self.is_data_incomplete = self.incomplete_check()
        self.last_column, self.null_index = self.last_model_response()

    def load_data(self):
        if self.df_exists:
            self.df = pd.read_csv(self.pd_file_path)

    def incomplete_check(self):
        if self.df is None:
            return False

        metrics_columns = [col for col in self.df.columns if col not in self.main_df_columns]

        for col in metrics_columns:
            column_data = self.df[col].tolist()
            column_data = [ast.literal_eval(data) if not pd.isna(data) else None for data in column_data ]

            for data in column_data:
                if data is None:
                    return True
                for score in data:
                    if score["extraction_needed"]:
                        return True
        return False

    def last_model_response(self):
        if self.df is None:
            return None, 0

        only_metrics = self.df.drop(columns=self.main_df_columns, errors='ignore')
        columns_len_nan = {}

        if len(only_metrics.columns) != 5:
            return None, 0

        for col in only_metrics.columns:
            column_data = self.df[col].tolist()
            for i, data in enumerate(column_data):
                if pd.isna(data) or data == "nan":
                    columns_len_nan[col] = i
                    break

        if len(columns_len_nan) == 0:
            return None, len(self.df)

        min_val = min(columns_len_nan.values())

        for key, value in columns_len_nan.items():
            if value == min_val:
                return key, value