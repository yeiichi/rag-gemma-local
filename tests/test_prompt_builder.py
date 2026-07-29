from rag_gemma_local.rag.prompt_builder import (
    answer_language_instruction,
    build_contextualize_prompt,
    build_qa_prompt,
    build_source_coverage_instruction,
)


def test_contextualize_prompt_returns_question_without_history() -> None:
    assert build_contextualize_prompt("What changed?", "") == "What changed?"


def test_qa_prompt_contains_context_and_question() -> None:
    prompt = build_qa_prompt("何が重要ですか？", "Content from [a.md]:\n重要です。")
    assert "Content from [a.md]" in prompt
    assert "何が重要ですか？" in prompt
    assert "Answer in Japanese." in prompt


def test_answer_language_instruction_defaults_to_english() -> None:
    assert answer_language_instruction("What matters?") == "Answer in English."


def test_source_coverage_instruction_names_multiple_sources() -> None:
    instruction = build_source_coverage_instruction(
        "Content from [english_policy.md]:\nA\n\n"
        "Content from [japanese_policy.md]:\nB"
    )

    assert "multiple source files" in instruction
    assert "english_policy.md" in instruction
    assert "japanese_policy.md" in instruction
    assert "at least one distinct point from each source file" in instruction
    assert "Do not answer from only one source file" in instruction
