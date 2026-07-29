from rag_gemma_local.llm.context import ContextOnlyLLM
from rag_gemma_local.rag.prompt_builder import build_qa_prompt


def test_context_llm_answers_from_first_context_block() -> None:
    prompt = build_qa_prompt(
        "What runtime is recommended?",
        "Content from [sample.md]:\nThe recommended runtime is llama.cpp.",
    )

    answer = ContextOnlyLLM().generate(prompt)

    assert "llama.cpp" in answer
    assert "Content from [sample.md]" in answer
