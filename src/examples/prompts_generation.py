import pandas as pd
import os

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from tqdm.auto import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_prompts(model_name: str,
                     prompts: list, temperature:float, main_path: str):

    model_file = model_name.replace(":", "_")
    temp_file = str(temperature).replace('.', '_')
    output_file = os.path.join(main_path, "data", "prompts", f"{model_file}_temp{temp_file}.csv")

    model = ChatOllama(model=model_name, temperature=temperature)
    columns = ["prompt_index", "prompt"] + [f"response_{i+1}" for i in range(3)]

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=columns)

    for i, prompt in tqdm(enumerate(prompts), desc=f"Generando prompts del modelo {model_name}", total=len(prompts)):
        row = df[df["prompt_index"] == i]

        if row.empty:
            new_row = {"prompt_index": i, "prompt": prompt}
            for r in range(3):
                new_row[f"response_{r+1}"] = ""
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_index = df.index[-1]
        else:
            row_index = row.index[0]

        for r in range(3):
            col_name = f"response_{r+1}"
            if pd.isna(df.at[row_index, col_name]) or df.at[row_index, col_name] == "":
                try:
                    response = model.invoke(prompt)
                except Exception as e:
                    response = f"Error: {str(e)}"

                if isinstance(response, AIMessage):
                    response = response.content

                df.at[row_index, col_name] = response
                df.to_csv(output_file, index=False)

if __name__ == "__main__":
    available_models = [
        "gpt-oss:20b",
        "llama3.1:8b",
        "mistral:latest",
        "phi3:14b",
        "qwen3:14b",
        "gemma3:12b",
        "deepseek-r1:32b",
    ]

    main_path = os.getcwd().split("src")[0]

    prompts = pd.read_csv(f"{main_path}/data/prompts_example.csv")["prompt"].tolist()
    with open(f"{main_path}data/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        for prompt in prompts
    ]

    temperatures = [0.5]

    os.makedirs(f"{main_path}/data/prompts", exist_ok=True)

    for model in available_models:
        for temp in temperatures:
            print(f"Generating prompts for model: {model} with temperature: {temp}")
            generate_prompts(model, messages, temp, main_path)