from fastapi import FastAPI
from models import ModelConfig, CompletionRequest, CompletionResponse

app = FastAPI(title="Unified LLM Gateway")


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
