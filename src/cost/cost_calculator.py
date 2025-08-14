import tiktoken
import ast
import pandas as pd
from src.examples.metrics import sanitize_thinking
from src.llms.prompts.relevance_prompts import relevance_user_prompt_gpt, relevance_user_prompt, relevance_system_prompt


def count_tokens(text: str, model: str = "gpt-4.1"):

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("error")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)

def df_token_count(df, model: str = "gpt-4.1") -> int:
    total_tokens = 0
    for column in df.columns:
        if df[column].dtype == 'object':
            for text in df[column].dropna():
                total_tokens += count_tokens(text, model)
    return total_tokens

def preprocess_text(text: str) -> str:
    if  isinstance(text, str):
        text = ast.literal_eval(text)
    return text

def preprocess_prompts(prompt):
    prompts = prompt["prompt"]
    prompts = preprocess_text(prompts)
    text = ""
    for prompt in prompts:
        if isinstance(prompt, dict):
            text += prompt.get("content", "")
        elif isinstance(prompt, str):
            text += prompt
    return text

def preprocess_responses(responses):
    for i, response in enumerate(responses):
        response = ast.literal_eval(response)
        for j, response_text in enumerate(response):
            response[j] = str(list(response_text.values()))
        responses[i] = response

    return responses


def preprocess_df(df):

    main_df_columns = [
        "prompt",
        "prompt_index",
    ]
    response_columns = [f"response_{i + 1}" for i in range(3)]

    prompts_df = df[response_columns]
    prompts_df = prompts_df.apply(sanitize_thinking)
    prompts_df = prompts_df.appy(preprocess_prompts)

    main_df_columns.extend(response_columns)

    evaluation_columns = [col for col in df.columns if col not in main_df_columns]
    evaluation_df = df[evaluation_columns]
    evaluation_df = evaluation_df.apply(preprocess_responses)


    for eval in evaluation_columns:
        evaluation_df[eval] = evaluation_df[eval].apply(lambda x: count_tokens(str(x)))

    return prompts_df, evaluation_df

if __name__ == "__main__":
    import os
    main = os.getcwd().split("src")[0]
    data_path = os.path.join(main, "data", "evaluated")
    data = "qwen3_14b_temp0_5_eval_gpt-oss_20b.csv"

    df = pd.read_csv(os.path.join(data_path, data))
    input_tokens, output_tokens = preprocess_df(df)

    print("Input Tokens:", input_tokens.sum())
    print("Output Tokens:", output_tokens.sum())
    print("Total output tokens:", output_tokens.sum().sum())

    cost_41_input = 0.00000125
    cost_41_output = 0.00001

    input_cost = input_tokens.sum() * cost_41_input
    output_cost = output_tokens.sum() * cost_41_output
    print("Cost of 41 input:", input_cost)
    print("Cost of 41 output:", output_cost)
    print("Total Cost:", input_cost.sum() + output_cost.sum())


