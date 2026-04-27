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
    token_count: int
    estimated_count: int
    total_tokens: int
    usd_cost: float
    model: str
    estimation_error: float


class TokenizeRequest(BaseModel):
    text: str
    model: str = "gpt-4o"


class TokenizeResponse(BaseModel):
    text: str
    token_count: int
    tokens: list[str]
    estimated_count: int
    estimation_error: float


class CompareRequest(BaseModel):
    text: str


class CompareResponse(BaseModel):
    text: str
    gpt4o_token_count: int
    gpt4o_tokens: list[str]
    gpt35_token_count: int
    gpt35_tokens: list[str]
    difference_percent: float


# ---- Asyncio calls --------------------------------
class BatchRequest(BaseModel):
    prompts: list[str] = Field(min_length=1)
    model: str = Field(default="llama3.2")
    system_prompt: Optional[str] = None


class BatchResponse(BaseModel):
    results: list[CompletionResponse]
    total_cost_usd: float
    total_time_ms: int
