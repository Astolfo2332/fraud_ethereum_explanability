from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import mlflow
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from difflib import SequenceMatcher

from src.llms.judges.utils.parsing.parsers import parse_json_response

embedding_model = SentenceTransformer("all-mpnet-base-v2")

class BaseScorer:
    def __init__(self, model, system_prompt:str, metric_name:str=None, structure=None, user_prompt:str=None):

        self.model = model
        self.metric_name = metric_name
        self.structure = structure
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.scores = []

    def score(self, responses: list, questions: list) -> list:
        model = ChatOllama(model=self.model, temperature=0.0)
        structure_model = model.with_structured_output(self.structure)

        eval_responses = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}\n\n {{'Answer to evaluate' : {response}}}"}
            ]

            score = structure_model.invoke(message)
            eval_responses.append(score)

        self.scores = eval_responses

        return self.scores

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

    def process_scores(self):
        numeric_scores = [score.score for score in self.scores]
        justifications = [score.justification for score in self.scores]

        if self.main_scorer:
            self.embeddings = embedding_model.encode(justifications)

        self.scores_cat = numeric_scores
        self.justifications = justifications

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

    def log_metrics(self):

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

        for score in self.scores_cat:
            if score is None:
                self.scores_num.append(np.nan)
            else:
                self.scores_num.append(self.categorical_to_numeric(score))

        self.scores_cat = []
        self.scores = []

        mlflow.log_metrics(
            {
                f"{self.metric_name}/mean": np.mean(self.scores_num),
                f"{self.metric_name}/std": np.std(self.scores_num),
                f"{self.metric_name}/var": np.var(self.scores_num)
            }
        )

class JudgeGPT(JudgeScorer):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None,
                 main_scorer: bool = False):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name,
                         main_scorer=main_scorer)
        self.structure = structure

    def score(self, responses: list, questions: list) -> list:
        model = ChatOllama(model=self.model, temperature=0.0)

        eval_responses = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}\n\n {{'Answer to evaluate' : {response}}}"}
            ]

            score = model.invoke(message)
            score = parse_json_response(score.content)
            eval_responses.append(score)

        self.scores = eval_responses

        return self.scores

class JudgeAccuracyGPT(JudgeScorer):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name)
        self.structure = structure

    def score(self, responses: list, questions: list) -> list:
        model = ChatOllama(model=self.model, temperature=0.0)

        eval_responses = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}"
        f"\n\n {{'input_numbers': {question} ,'Answer to evaluate' : {response}}}"}
            ]

            score = model.invoke(message)
            score = parse_json_response(score.content)
            eval_responses.append(score)

        self.scores = eval_responses

        return self.scores

class JudgeAccuracy(JudgeScorer):
    def __init__(self, model:str, system_prompt:str,
                 user_prompt:str, structure=None, metric_name:str=None):
        super().__init__(model=model, system_prompt=system_prompt,
                         user_prompt=user_prompt, metric_name=metric_name)
        self.structure = structure

    def score(self, responses: list, questions: list) -> list:
        model = ChatOllama(model=self.model, temperature=0.0)
        structure_model = model.with_structured_output(self.structure)

        eval_responses = []

        for response, question in zip(responses , questions):
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"# Instructions:\n\n{self.user_prompt}"
        f"\n\n {{'input_numbers': {question} ,'Answer to evaluate' : {response}}}"}
            ]

            score = structure_model.invoke(message)
            eval_responses.append(score)

        self.scores = eval_responses

        return self.scores
