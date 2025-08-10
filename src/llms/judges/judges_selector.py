from src.llms.judges.judges import JudgeScorer, JudgeGPT, JudgeAccuracy, JudgeAccuracyGPT
from src.llms.prompts.correctness_prompts import correctness_system_prompt, correctness_user_prompt, correctness_user_prompt_gpt
from src.llms.prompts.relevance_prompts import relevance_user_prompt_gpt, relevance_user_prompt, relevance_system_prompt
from src.llms.prompts.accuracy_prompts import accuracy_user_prompt, accuracy_system_prompt, accuracy_user_prompt_gpt
from src.llms.prompts.completness_prompt import completeness_system_prompt, completeness_user_prompt, completeness_user_prompt_gpt


def correctness_judge(model: str, structure, is_gpt: bool = False) -> JudgeScorer:

    metric_name = model + "_correctness"
    metric_name = metric_name.replace(":", "_")

    system_prompt = correctness_system_prompt
    user_prompt = correctness_user_prompt_gpt if is_gpt else correctness_user_prompt

    if  is_gpt:
        return JudgeGPT(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name,
                        main_scorer=True)


    return JudgeScorer(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name,
                       main_scorer=True)

def relevance_judge(model: str, structure, is_gpt: bool = False) -> JudgeScorer:

    metric_name = model + "_relevance"
    metric_name = metric_name.replace(":", "_")

    system_prompt = relevance_system_prompt
    user_prompt = relevance_user_prompt_gpt if is_gpt else relevance_user_prompt

    if  is_gpt:
        return JudgeGPT(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name)


    return JudgeScorer(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name)

def accuracy_judge(model: str, structure, is_gpt: bool = False) -> JudgeScorer:
    metric_name = model + "_accuracy"
    metric_name = metric_name.replace(":", "_")

    system_prompt = accuracy_system_prompt
    user_prompt = accuracy_user_prompt_gpt if is_gpt else accuracy_user_prompt

    if  is_gpt:
        return JudgeAccuracyGPT(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name)


    return JudgeAccuracy(model=model,
                          system_prompt=system_prompt,
                          user_prompt=user_prompt,
                          structure=structure,
                          metric_name=metric_name)

def completeness_judge(model: str, structure, is_gpt: bool = False) -> JudgeScorer:
    metric_name = model + "_completeness"
    metric_name = metric_name.replace(":", "_")

    system_prompt = completeness_system_prompt
    user_prompt = completeness_user_prompt_gpt if is_gpt else completeness_user_prompt

    if  is_gpt:
        return JudgeGPT(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name)


    return JudgeScorer(model=model,
                       system_prompt=system_prompt,
                       user_prompt=user_prompt,
                       structure=structure,
                       metric_name=metric_name)


def get_all_metrics(model: str, structure, is_gpt: bool = False) -> list[JudgeScorer]:
    return [
        correctness_judge(model, structure, is_gpt),
        relevance_judge(model, structure, is_gpt),
        accuracy_judge(model, structure, is_gpt),
        completeness_judge(model, structure, is_gpt)
    ]