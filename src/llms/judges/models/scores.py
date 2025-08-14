from pydantic import BaseModel, Field

class JudgeOutput(BaseModel):
    justification: str
    score: str

class ScoreCorrection(BaseModel):
    score: str
