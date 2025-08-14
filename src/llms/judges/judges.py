from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import mlflow
import numpy as np
import pandas as pd
import ast

from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from difflib import SequenceMatcher
import concurrent.futures

from src.llms.judges.models.scores import JudgeOutput
from src.llms.judges.utils.parsing.parsers import parse_json_response, llm_format, correct_llm_score
from src.llms.managers.ollama_manager import ollama_manager

embedding_model = SentenceTransformer("all-mpnet-base-v2")

def terminal_invoke(model, prompt, timeout=300):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.invoke, prompt)
            try:
                response = future.result(timeout=timeout)
                return response
            except concurrent.futures.TimeoutError:
                print("La llamada al modelo ha excedido el tiempo límite.")
                ollama_manager.quick_restart()
                return JudgeOutput(justification="La llamada al modelo ha excedido el tiempo límite.", score="Bad")

class BaseScorer:
    def __init__(self, model, system_prompt:str, metric_name:str=None, structure=None, user_prompt:str=None):

        self.model = model
        self.metric_name = metric_name
        self.structure = structure
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.scores = []


    def categorical_to_numeric(self, score: str) -> float:
        score_mapping = {
            "Bad": 1.0,
            "Regular": 2.0,
            "Well": 3.0,
            "Good": 4.0,
            "Excellent": 5.0
        }

        score = score.strip().replace("*", "")

        return score_mapping.get(score, np.nan)


    def score(self, responses: list, questions: list) -> dict:
        model = ChatOllama(model=self.model, temperature=0.0)
        structure_model = model.with_structured_output(self.structure)

        eval_responses = []
        scores_dict = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}\n\n {{'Answer to evaluate' : {response}}}"}
            ]

            score = terminal_invoke(structure_model, message)
            eval_responses.append(score)

            if self.categorical_to_numeric(score.score) is np.nan:
                score.score = correct_llm_score(score)

            scores_dict.append({
                "extraction_needed": False,
                "score": score.score,
                "justification": score.justification
            })

        column = {
            "column_name": self.metric_name,
            "data": scores_dict
        }

        self.scores = eval_responses

        return column

    def log_metrics(self, *args):
        NotImplementedError("Este método es propio de la subclase, ya que dependerá de la información de las metricas")

class JudgeScorer(BaseScorer):
    def __init__(self, model:str, system_prompt:str, user_prompt:str, structure=None,
                 metric_name:str=None, main_scorer: bool = False):

        super().__init__(model=model, system_prompt=system_prompt,
                         metric_name=metric_name, structure=structure,
                         user_prompt=user_prompt)

        self.embeddings = None
        self.main_scorer = main_scorer
        self.scores_cat = []
        self.scores_num = []
        self.justifications = []
        self.metric_name = metric_name
        self.cosine_sim = []
        self.levenshtein_sim = []
        self.diff_sim = []

    def get_all_matrix(self):
        return self.levenshtein_sim, self.diff_sim, self.cosine_sim

    def process_scores(self):
        self.scores = [score for score in self.scores if score is not None]

        if not self.scores:
            return

        numeric_scores = [score.score for score in self.scores]
        justifications = [score.justification for score in self.scores]

        if self.main_scorer:
            self.embeddings = embedding_model.encode(justifications)

        self.scores_cat = numeric_scores
        self.justifications = justifications

        for score in self.scores_cat:
            if score is None:
                self.scores_num.append(np.nan)
            else:
                self.scores_num.append(self.categorical_to_numeric(score))



    def make_embeddings_metrics(self):
        if self.embeddings is None:
            raise ValueError("Debes ejecutar process_scores antes de calcular las métricas de embeddings.")

        distance_matrix = cosine_similarity(self.embeddings)

        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
        upper_tri_values = distance_matrix[upper_tri_indices]

        return upper_tri_values


    def make_similarity_metrics(self):

        n = len(self.scores_cat)

        lev_matrix = np.zeros((n, n))
        diff_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if  i<= j:
                    lev_matrix[i, j] = levenshtein_ratio(self.justifications[i], self.justifications[j])
                    diff_matrix[i, j] = SequenceMatcher(None, self.justifications[i], self.justifications[j]).ratio()

        upper_tri_indices = np.triu_indices_from(lev_matrix, k=1)

        upper_tri_lev = lev_matrix[upper_tri_indices]
        upper_tri_diff = diff_matrix[upper_tri_indices]

        return upper_tri_lev, upper_tri_diff

    def log_metrics(self) -> dict:

        if self.main_scorer:
            upper_tri_lev, upper_tri_diff = self.make_similarity_metrics()
            cosine_sim = self.make_embeddings_metrics()

            self.cosine_sim.append(cosine_sim)
            self.levenshtein_sim.append(upper_tri_lev)
            self.diff_sim.append(upper_tri_diff)

            mlflow.log_metrics({
                "cosine_similarity/mean": np.mean([np.mean(sim) for sim in self.cosine_sim]),
                "cosine_similarity/std": np.mean([np.std(sim) for sim in self.cosine_sim]),
                "cosine_similarity/var": np.mean([np.var(sim) for sim in self.cosine_sim]),
                "levenshtein_similarity/mean": np.mean([np.mean(sim) for sim in self.levenshtein_sim]),
                "levenshtein_similarity/std": np.mean([np.std(sim) for sim in self.levenshtein_sim]),
                "levenshtein_similarity/var": np.mean([np.var(sim) for sim in self.levenshtein_sim]),
                "diff_similarity/mean": np.mean([np.mean(sim) for sim in self.diff_sim]),
                "diff_similarity/std": np.mean([np.std(sim) for sim in self.diff_sim]),
                "diff_similarity/var": np.mean([np.var(sim) for sim in self.diff_sim])
            })



        self.scores_cat = []
        self.scores = []

        mlflow.log_metrics(
            {
                f"{self.metric_name}/mean": np.mean(self.scores_num),
                f"{self.metric_name}/std": np.std(self.scores_num),
                f"{self.metric_name}/var": np.var(self.scores_num)
            }
        )
        return {"data": self.scores_num, "column_name": self.metric_name}

class JudgeGPT(JudgeScorer):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None,
                 main_scorer: bool = False):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name,
                         main_scorer=main_scorer)
        self.structure = structure
        self.need_extraction = []

    def score(self, responses: list, questions: list) -> dict:
        model = ChatOllama(model=self.model, temperature=0.0)

        eval_responses = []
        scores_dict = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}\n\n {{'Answer to evaluate' : {response}}}"}
            ]

            score = terminal_invoke(model, message)
            if isinstance(score, JudgeOutput):
                score = AIMessage(content="La llamada exedió el tiempo límite")

            score, no_need_extraction = parse_json_response(score.content)

            if no_need_extraction:
                eval_responses.append(score)
            else:
                self.need_extraction.append(score)

            if no_need_extraction:
                scores_dict.append({
                    "extraction_needed": False,
                    "score": score.score,
                    "justification": score.justification
                })
            else:
                scores_dict.append({
                    "extraction_needed": True,
                    "data": score
                })

        column = {
            "column_name": self.metric_name,
            "data": scores_dict
        }

        self.scores = eval_responses

        return column


    def finish(self, df:pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.main_scorer:
            self.main_scorer = False

        column_data = df[self.metric_name].tolist()

        for i, data in tqdm(enumerate(column_data), desc=f"Procesando filas con datos sin extracción en {self.metric_name}", total=len(column_data)):
            if  isinstance(data, str):
                data = ast.literal_eval(data)

            for j, score in enumerate(data):
                if not score["extraction_needed"]:
                    continue

                score_data = score["data"]

                #TODO: Reevaluar las primeras corridas ya que hay datos None que no se pudieron procesar

                if score_data is None:
                    continue

                score_structure = llm_format(score_data)

                if self.categorical_to_numeric(score_structure.score) is np.nan:
                    score_structure.score = correct_llm_score(score_structure)

                data[j] = {
                    "extraction_needed": False,
                    "score": score_structure.score,
                    "justification": score_structure.justification
                }
                self.scores.append(score_structure)
                df.at[i, self.metric_name] = data

        self.process_scores()
        log_metric = self.log_metrics()

        return df, log_metric


class JudgeAccuracyGPT(JudgeGPT):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name)
        self.structure = structure
        self.need_extraction = []

    def score(self, responses: list, questions: list) -> dict:
        model = ChatOllama(model=self.model, temperature=0.0)

        eval_responses = []

        scores_dict = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}"
        f"\n\n {{'input_numbers': {question} ,'Answer to evaluate' : {response}}}"}
            ]

            score = terminal_invoke(model, message)

            if isinstance(score, JudgeOutput):
                score = AIMessage(content="La llamada exedió el tiempo límite")

            score, no_need_extraction = parse_json_response(score.content)
            if no_need_extraction:
                eval_responses.append(score)
            else:
                self.need_extraction.append(score)

            if no_need_extraction:
                scores_dict.append({
                    "extraction_needed": False,
                    "score": score.score,
                    "justification": score.justification
                })
            else:
                scores_dict.append({
                    "extraction_needed": True,
                    "data": score
                })

        column = {
            "column_name": self.metric_name,
            "data": scores_dict
        }

        self.scores = eval_responses

        return column


class JudgeAccuracy(JudgeScorer):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name)
        self.structure = structure

    def score(self, responses: list, questions: list) -> dict:
        model = ChatOllama(model=self.model, temperature=0.0)
        structure_model = model.with_structured_output(self.structure)

        eval_responses = []
        scores_dict = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}"
        f"\n\n {{'input_numbers': {question} ,'Answer to evaluate' : {response}}}"}
            ]

            score = terminal_invoke(structure_model, message)
            eval_responses.append(score)


            scores_dict.append({
                    "extraction_needed": False,
                    "score": score.score,
                    "justification": score.justification
                })

        column = {
            "column_name": self.metric_name,
            "data": scores_dict
        }

        self.scores = eval_responses

        return column
