from src.main import tokenize_text, build_messages


def test_tokenize_text_normal():
    num_tokens, decoded_tokens = tokenize_text("Hello world")
    assert isinstance(num_tokens, int)
    assert num_tokens > 0
    assert len(decoded_tokens) > 0


def test_tokenize_text_empty():
    num_tokens, decoded_tokens = tokenize_text("")
    assert num_tokens == 0
    assert len(decoded_tokens) == 0


def test_build_messages_no_system():
    results = build_messages("hola", None)
    assert len(results) == 1
    assert results[0]["role"] == "user"
    assert results[0]["content"] == "hola"


def test_build_messages_normal():
    results = build_messages("hola", "eres un asistente")
    assert len(results) == 2
    assert results[0]["role"] == "system"
    assert results[1]["role"] == "user"
