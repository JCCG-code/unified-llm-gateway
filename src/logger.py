import logging
from src.models import CompletionRequest, CompletionResponse

# Initializations
logger = logging.getLogger("gateway")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
)
logger.addHandler(handler)
logger.propagate = False


def log_request(content: CompletionRequest | CompletionResponse):
    if isinstance(content, CompletionRequest):
        logger.info(f"REQ  | model={content.model} prompt='{content.prompt}'")
    else:
        logger.info(
            f"RES | model={content.model.name} tokens_in={content.input_tokens} tokens_out={content.output_tokens} cost=${content.cost_usd:.2f} time={content.response_time_ms}ms"
        )
