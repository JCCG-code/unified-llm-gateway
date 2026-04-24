from pydantic import BaseModel


# Completion request /complete endpoint
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False


# Completion response /complete endpoint
class CompletionResponse(BaseModel):
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
