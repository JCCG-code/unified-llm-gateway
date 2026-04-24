from fastapi import FastAPI
from models import CompletionRequest, CompletionResponse

app = FastAPI(title="Unified LLM Gateway")


@app.get("/health")
async def health() -> CompletionResponse:
    return CompletionResponse(
        model="MiniMax-M2.7",
        content="Hola que tal",
        input_tokens=10,
        output_tokens=10,
        cost_usd=1.1,
    )


@app.post("/complete")
async def complete(req: CompletionRequest) -> CompletionResponse:
    return CompletionResponse(
        model=req.model,
        content=f"Respuesta simulada para: {req.prompt}",
        # Contar tokens de req.prompt
        input_tokens=10,
        output_tokens=10,
        cost_usd=1.1,
    )
