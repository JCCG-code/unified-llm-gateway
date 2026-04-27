from pydantic import ValidationError
import pytest
from src.models import CompletionRequest, ModelConfig


def test_completion_request_defaults():
    req = CompletionRequest(prompt="hola")
    assert req.model == "llama3.2"
    assert req.max_tokens == 1000
    assert req.temperature == 0.7
    assert not req.stream


def test_completion_request_validation():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="hola", temperature=2.1)
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="hola", max_tokens=5000)


def test_model_config():
    model = ModelConfig(name="llama3.2", cost_input_token=0.25, cost_output_token=0.5)
    assert model.name == "llama3.2"
    assert model.cost_input_token == 0.25
    assert model.cost_output_token == 0.5
