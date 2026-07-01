from src.prompt import build_rag_prompt


def test_build_rag_prompt_includes_context():
    context = "Sample medical context"
    prompt = build_rag_prompt(context)

    assert "You are a medical assistant" in prompt
    assert context in prompt
    assert "Context:" in prompt
