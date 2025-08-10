import json
from langchain_ollama import ChatOllama

from src.llms.judges.models.scores import JudgeOutput

system_prompt = """Eres un especialista de extracción de texto
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

def parse_json_response(response: str):

    response = response.split("```json")

    if len(response) < 2:
        response = llm_format(response)
        return response

    response = response[1]

    if not response:
        return None
    response = response.split("```")[0].strip()

    try:
        response = json.loads(response)
        res = JudgeOutput(**response)
        return res
    except json.JSONDecodeError:
        return None

def llm_format(response: str):
    model = ChatOllama(model="gemma3:12b", temperature=0.0)
    model_with_structure = model.with_structured_output(JudgeOutput)
    response = model_with_structure.invoke(system_prompt + response[0])
    return response
