import os
import mlflow
import pandas as pd
import ast
from tqdm.auto import tqdm

from src.llms.judges.models.scores import JudgeOutput
from src.llms.judges.judges_selector import get_all_metrics

mlflow.langchain.autolog()

judges_models = {
    "gpt-oss:20b": True,
    "gemma3:12b": False,
}

main_path = os.getcwd().split("src")[0]

data_path = f"{main_path}/data/prompts"

available_data = os.listdir(data_path)

def make_a_score(score_fn, responses, original_prompt):
    scores = score_fn.score(responses, original_prompt)
    score_fn.process_scores()
    score_fn.log_metrics()

def finalized_all_metrics(all_metrics):
    for metric in all_metrics:
        metric.finish()

def sanitize_thinking(response:str) -> str:
    if "<think>" in response:
        return response.split("</think>")[1]
    return response

def all_metrics():

    mlflow.set_experiment("llms_evaluation")

    for model, is_gpt in judges_models.items():
        phbar = tqdm(available_data, total=len(available_data), desc="")
        for data in phbar:

            all_metrics = get_all_metrics(model, JudgeOutput, is_gpt)
            phbar.set_description(f"Evaluando {data} con modelo {model}")
            with mlflow.start_run(run_name=data.replace(".csv", "")):
                df = pd.read_csv(f"{data_path}/{data}")
                for row in tqdm(df.iterrows(), desc="Analizando filas", total=len(df)):
                    original_prompt = row[1]["prompt"]
                    original_prompt = ast.literal_eval(original_prompt)
                    original_prompt = original_prompt[1]["content"]

                    responses = [sanitize_thinking(row[1][f"response_{i+1}"]) for i in range(3)]
                    original_prompt = [original_prompt] * len(responses)

                    for metric in all_metrics:
                        make_a_score(metric, responses, original_prompt)
                if is_gpt:
                    finalized_all_metrics(all_metrics)

if __name__ == "__main__":
    all_metrics()


