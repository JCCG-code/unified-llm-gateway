from fastapi import FastAPI, HTTPException
from models import ModelConfig, CompletionRequest, CompletionResponse, CostEstimate
from ollama import AsyncClient


# Constant variables
AVAILABLE_MODELS = [
    ModelConfig(name="llama3.2", cost_input_token=0.25, cost_output_token=0.5),
    ModelConfig(name="gemma4-e4b", cost_input_token=0.25, cost_output_token=0.5),
]


# Initializations
app = FastAPI(title="Unified LLM Gateway")
ollama_client = AsyncClient()


# Helpers
def build_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


@app.get("/health")
async def health() -> CompletionResponse:
    return CompletionResponse(
        model=ModelConfig(
            name="MiniMax-M2.7", cost_input_token=0.05, cost_output_token=0.1
        ),
        content="Hola que tal",
        input_tokens=10,
        output_tokens=10,
        cost_usd=1.1,
        response_time_ms=203,
    )


@app.post("/complete")
async def complete(req: CompletionRequest) -> CompletionResponse:
    # Searchs and extracts requested model
    model = next(m for m in AVAILABLE_MODELS if req.model == m.name)
    # Constructs an array of models to try
    models_to_try = [model] + [m for m in AVAILABLE_MODELS if m.name != model.name]
    # Detects and builds array message
    messages = build_messages(req.prompt, req.system_prompt)
    # Model call iteratively
    for m in models_to_try:
        try:
            responseChat = await ollama_client.chat(model=m.name, messages=messages)
            # Extract data
            content = responseChat.message.content
            input_tokens = responseChat.prompt_eval_count
            output_tokens = responseChat.eval_count
            total_duration = responseChat.total_duration
            # Checks values
            if (
                content is None
                or input_tokens is None
                or output_tokens is None
                or total_duration is None
            ):
                continue
            # Calc cost
            cost = (
                input_tokens * m.cost_input_token + output_tokens * m.cost_output_token
            )
            # Calc response time in miliseconds
            response_time_ms = total_duration // 1_000_000
            # Return statement
            return CompletionResponse(
                model=m,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                response_time_ms=response_time_ms,
            )
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="No model available")


@app.get("/models")
async def models() -> list[ModelConfig]:
    return AVAILABLE_MODELS


@app.get("/models/{name}")
async def model_by_name(name: str) -> ModelConfig:
    try:
        return next(m for m in AVAILABLE_MODELS if name == m.name)
    except StopIteration:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/estimate-cost")
async def estimate_cost(req: CompletionRequest) -> CostEstimate:
    # Tokens count extract
    prompt_count = len(req.prompt)
    tokens_count = prompt_count // 4
    # Estimated cost
    modelEl = next(m for m in AVAILABLE_MODELS if req.model == m.name)
    usdCost = modelEl.cost_input_token * tokens_count
    # Return statement
    return CostEstimate(
        estimated_input_tokens=tokens_count,
        estimated_cost_usd=usdCost,
        model=modelEl.name,
    )
