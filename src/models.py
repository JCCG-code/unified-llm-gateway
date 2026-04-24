from pydantic import BaseModel, AfterValidator, Field
from datetime import datetime
from typing import Annotated


# Model configuration interface
class ModelConfig(BaseModel):
    name: str
    cost_input_token: float
    cost_output_token: float


# Checks a valid temperature
def valid_temperature(value: float) -> float:
    if value < 0 or value > 2:
        raise ValueError("No es una temperatura valida")
    return value


# Completion request /complete endpoint
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1000
    temperature: Annotated[float, AfterValidator(valid_temperature)]
    stream: bool = False


# Completion response /complete endpoint
class CompletionResponse(BaseModel):
    model: ModelConfig
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: datetime = Field(default_factory=datetime.now)
