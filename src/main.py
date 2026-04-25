from fastapi import FastAPI, HTTPException
from models import ModelConfig, CompletionRequest, CompletionResponse, CostEstimate
from ollama import AsyncClient
from logger import log_request

# Constant variables
AVAILABLE_MODELS = [
    ModelConfig(name="llama3.2", cost_input_token=0.25, cost_output_token=0.5),
    ModelConfig(name="gemma4-e4b", cost_input_token=0.25, cost_output_token=0.5),
]


# Initializations
app = FastAPI(title="Unified LLM Gateway")
ollama_client = AsyncClient()


# Helpers
def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
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
    # Logger request
    log_request(req)
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
            # Res statement
            res = CompletionResponse(
                model=m,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                response_time_ms=response_time_ms,
            )
            log_request(res)
            # Return statement
            return res
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="No model available")


@app.get("/models")
async def models() -> list[ModelConfig]:
    return AVAILABLE_MODELS


@app.get("/models/{name}")
async def get_model_by_name(name: str) -> ModelConfig:
    try:
        return next(m for m in AVAILABLE_MODELS if name == m.name)
    except StopIteration:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/models", status_code=201)
async def new_model(req: ModelConfig) -> ModelConfig:
    model = next((m for m in AVAILABLE_MODELS if req.name == m.name), None)
    if model:
        raise HTTPException(status_code=409, detail="Model already exists")
    AVAILABLE_MODELS.append(req)
    return req


@app.put("/models/{name}")
async def update_mode(name: str, req: ModelConfig) -> ModelConfig:
    index = next((i for i, m in enumerate(AVAILABLE_MODELS) if name == m.name), None)
    if index is None:
        raise HTTPException(status_code=404, detail="Model not found")
    AVAILABLE_MODELS[index] = req
    return AVAILABLE_MODELS[index]


@app.delete("/models/{name}", status_code=204)
async def delete_model_by_name(name: str) -> None:
    try:
        model = next(m for m in AVAILABLE_MODELS if name == m.name)
        AVAILABLE_MODELS.remove(model)
    except StopIteration:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/estimate-cost")
async def estimate_cost(req: CompletionRequest) -> CostEstimate:
    # Tokens count extract
    prompt_count = len(req.prompt)
    tokens_count = prompt_count // 4
    # Estimated cost
    model = next(m for m in AVAILABLE_MODELS if req.model == m.name)
    usd_cost = model.cost_input_token * tokens_count
    # Return statement
    return CostEstimate(
        estimated_input_tokens=tokens_count,
        estimated_cost_usd=usd_cost,
        model=model.name,
    )
