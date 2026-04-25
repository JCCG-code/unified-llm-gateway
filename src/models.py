from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# Model configuration interface
class ModelConfig(BaseModel):
    name: str
    cost_input_token: float
    cost_output_token: float


# Completion request /complete endpoint
class CompletionRequest(BaseModel):
    model: str = Field(default="llama3.2")
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    system_prompt: Optional[str] = None


# Completion response /complete endpoint
class CompletionResponse(BaseModel):
    model: ModelConfig
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    response_time_ms: int
    created_at: datetime = Field(default_factory=datetime.now)


# Cost estimated model
class CostEstimate(BaseModel):
    estimated_input_tokens: int
    estimated_cost_usd: float
    model: str
