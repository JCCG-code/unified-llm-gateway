from fastapi import FastAPI
from models import ModelConfig, CompletionRequest, CompletionResponse, CostEstimate
# from ollama import chat


app = FastAPI(title="Unified LLM Gateway")

# Constant variables
AVAILABLE_MODELS = [
    ModelConfig(name="llama3.2", cost_input_token=0.25, cost_output_token=0.5),
    ModelConfig(name="gemma4-e4b", cost_input_token=0.25, cost_output_token=0.5),
]


# response = chat(
#     model="llama3.2",
#     messages=[{"role": "user", "content": "Hello!"}],
# )
# print(response.message.content)


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
    # Requested model
    model = ModelConfig(name=req.model, cost_input_token=0.05, cost_output_token=0.1)
    input_tokens = 10
    output_tokens = 10
    # Return statement
    return CompletionResponse(
        model=model,
        content=f"Respuesta simulada para: {req.prompt}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=model.cost_input_token * input_tokens
        + model.cost_output_token * output_tokens,
    )


@app.get("/models")
async def models() -> list[ModelConfig]:
    return AVAILABLE_MODELS


@app.post("/estimate-cost")
async def estimate_cost(req: CompletionRequest) -> CostEstimate:
    # Tokens count extract
    promptCount = len(req.prompt)
    tokensCount = int(promptCount / 4)
    # Estimated cost
    modelEl = next(m for m in AVAILABLE_MODELS if req.model == m.name)
    usdCost = modelEl.cost_input_token * tokensCount
    # Return statement
    return CostEstimate(
        estimated_input_tokens=tokensCount,
        estimated_cost_usd=usdCost,
        model=modelEl.name,
    )
