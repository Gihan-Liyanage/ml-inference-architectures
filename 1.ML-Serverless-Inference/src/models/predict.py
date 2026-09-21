from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    displacement: float = Field(..., gt=0)
    horsepower: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    acceleration: float = Field(..., gt=0)
    model_year: int = Field(..., ge=0, le=99)
    origin: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predicted_mpg: float
    input: PredictRequest
