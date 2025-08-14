import json
import os
import time

import mlflow
import numpy as np
import pandas as pd
import ast

from ollama import ResponseError
from tqdm.auto import tqdm

from src.llms.judges.models.scores import JudgeOutput
from src.llms.judges.judges_selector import get_all_metrics

from src.llms.managers.ollama_manager import ollama_manager
from src.llms.managers.file_manager import FileManager

mlflow.langchain.autolog()

judges_models = {
    "gpt-oss:20b": True,
    "gemma3:12b": False,
}

main_path = os.getcwd().split("src")[0]

data_path = f"{main_path}/data/test"
process_data_path = f"{main_path}/data/evaluated"
os.makedirs(process_data_path, exist_ok=True)

available_data = os.listdir(data_path)

def make_a_score(score_fn, responses, original_prompt):
    scores = score_fn.score(responses, original_prompt)
    score_fn.process_scores()
    log_metric = score_fn.log_metrics()
    return scores, log_metric

def finalized_all_metrics(all_metrics, df):
    metrics = {}
    for metric in all_metrics:
        df, log_metric = metric.finish(df)
        metrics[log_metric["column_name"]] = log_metric["data"]
    return df, metrics

def load_metrics(all_metrics, metric_path):
    if os.path.exists(metric_path):
        with open(metric_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        for metric in all_metrics:
            if metric.main_scorer:
                metric.levenshtein_sim = metrics.get("levenshtein_matrix", [])
                metric.diff_sim = metrics.get("diff_matrix", [])
                metric.cosine_sim = metrics.get("cosine_matrix", [])

                metric.levenshtein_sim = list_to_numpy(metric.levenshtein_sim)
                metric.diff_sim = list_to_numpy(metric.diff_sim)
                metric.cosine_sim = list_to_numpy(metric.cosine_sim)

            if metric.metric_name in metrics:
                metric.scores_num = metrics[metric.metric_name]

def list_to_numpy(data:list) -> list:
    return [np.array(item) for item in data]

def sanitize_thinking(response:str) -> str:
    if "<think>" in response:
        return response.split("</think>")[1]
    return response

def all_metrics():

    mlflow.set_experiment("llms_evaluation_test")

    for model, is_gpt in judges_models.items():
        phbar = tqdm(available_data, total=len(available_data), desc="")
        for data in phbar:
            all_metrics = get_all_metrics(model, JudgeOutput, is_gpt)
            phbar.set_description(f"Evaluando {data} con modelo {model}")

            with mlflow.start_run(run_name=data.replace(".csv", "")):
                df = pd.read_csv(f"{data_path}/{data}")
                process_df = df.copy()

                metrics = {}
                metric_path = f"{process_data_path}/{data.replace('.csv', '_metrics')}" + model.replace(":", "_") + ".json"
                process_df_name = data.replace(".csv", "") + "_eval_" + model.replace(":", "_") + ".csv"
                process_df_path = f"{process_data_path}/{process_df_name}"

                file_manager = FileManager(process_df_path, metric_path)

                if file_manager.df_exists:
                    load_metrics(all_metrics, metric_path)
                    process_df = file_manager.df
                    print("Iniciando desde:", file_manager.null_index)

                for row in tqdm(df.iterrows(), desc="Analizando filas", total=len(df)):
                    idx = row[0]

                    if idx < file_manager.null_index:
                        continue

                    original_prompt = row[1]["prompt"]
                    original_prompt = ast.literal_eval(original_prompt)
                    original_prompt = original_prompt[1]["content"]

                    responses = [sanitize_thinking(row[1][f"response_{i+1}"]) for i in range(3)]
                    original_prompt = [original_prompt] * len(responses)

                    for metric in all_metrics:
                        scores, log_metric = make_a_score(metric, responses, original_prompt)
                        if metric.main_scorer:
                            lev_mat, diff_mat, cos_mat = metric.get_all_matrix()
                            metrics["levenshtein_matrix"] = [data.tolist() for data in lev_mat]
                            metrics["diff_matrix"] = [data.tolist() for data in diff_mat]
                            metrics["cosine_matrix"] = [data.tolist() for data in cos_mat]

                        metrics[log_metric["column_name"]] = log_metric["data"]

                        with open(metric_path, "w", encoding="utf-8") as f:
                            json.dump(metrics, f)

                        if scores["column_name"] not in process_df.columns:
                            process_df[scores["column_name"]] = None

                        process_df.at[row[0], scores["column_name"]] = scores["data"]
                        process_df.to_csv(f"{process_data_path}/{process_df_name}", index=False)

                if is_gpt or (file_manager.df_exists and file_manager.is_data_incomplete):
                    df, metrics = finalized_all_metrics(all_metrics, process_df)
                    df.to_csv(f"{process_data_path}/{process_df_name}", index=False)
                    with open(metric_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f)

if __name__ == "__main__":
    ollama_manager.start_ollama()
    while True:
        try:
            all_metrics()
        except ResponseError as e:
            print(f"Error: {e}")
            ollama_manager.quick_restart()