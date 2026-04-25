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
    )


@app.post("/complete")
async def complete(req: CompletionRequest) -> CompletionResponse:
    # Searchs and extracts requested model
    model = next(m for m in AVAILABLE_MODELS if req.model == m.name)
    # Detects and builds array message
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})
    # Model call
    responseChat = await ollama_client.chat(model=model.name, messages=messages)
    # Extract data
    content = responseChat.message.content
    input_tokens = responseChat.prompt_eval_count
    output_tokens = responseChat.eval_count
    # Checks values
    if content is None or input_tokens is None or output_tokens is None:
        raise HTTPException(status_code=500, detail="Error model response")
    # Calc cost
    cost = (
        input_tokens * model.cost_input_token + output_tokens * model.cost_output_token
    )
    # Return statement
    return CompletionResponse(
        model=model,
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )


@app.get("/models")
async def models() -> list[ModelConfig]:
    return AVAILABLE_MODELS


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
