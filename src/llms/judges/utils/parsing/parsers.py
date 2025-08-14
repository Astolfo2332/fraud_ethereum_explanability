import json
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.llms.judges.models.scores import JudgeOutput, ScoreCorrection
from transformers import pipeline

system_prompt = """
Eres un especialista de extracción de texto
Tu tarea es extraer la información del siguiente texto.
Ten en cuenta que para el apartado score solo existen 5 clasificaciones:
- Bad
- Regular
- Well
- Good
- Excellent

La justificación debe ser extraida tal y como se encuentra en el texto, sin modificarla.
El texto a extraer es el siguiente:

"""

system_prompt_correction = """
**You are a text extraction specialist.**  
Your task is to identify and return **only** the correct rating based on the provided justification.  

The only possible ratings are:  
- Bad
- Regular
- Well
- Good
- Excellent

**Instructions:**  
1. Analyze **only** the following justification to determine the correct rating:
`{justification}`  
2. Keep in mind that the previously assigned incorrect rating was: `{score}`
3. Return **only** one of the 5 listed ratings, with **no** additional text, explanations, extra punctuation, or words outside the list.

"""

template_text = """
The model justification is: {justification}
The model score is: {score}
"""

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def parse_json_response(response: str):

    if not response:
        return None, False

    response = response.split("```json")

    if len(response) < 2:
        return response, False

    response = response[1]
    response = response.split("```")[0].strip()

    try:
        response = json.loads(response)
        res = JudgeOutput(**response)
        return res, True
    except json.JSONDecodeError:
        return response, False
    except ValidationError:
        return response, False

def correct_llm_score(score: JudgeOutput):
    text = template_text.format(justification=score.justification, score=score.score)
    response = classifier(text, candidate_labels=["Bad", "Regular", "Well", "Good", "Excellent"], multi_label=False)
    response = response["labels"][0]
    print("Original Score:", score.score, "Corrected Score:", response)

    return response


def llm_format(response):
    model = ChatOllama(model="gemma3:12b-it-qat", temperature=0.0)
    model_with_structure = model.with_structured_output(JudgeOutput)
    response = model_with_structure.invoke(system_prompt + response[0])
    return response
