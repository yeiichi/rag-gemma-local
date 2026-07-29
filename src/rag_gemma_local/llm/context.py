from __future__ import annotations

import re


class ContextOnlyLLM:
    """Deterministic provider for smoke tests before a real local LLM is ready."""

    def generate(self, prompt: str) -> str:
        if "Standalone query:" in prompt:
            return self._latest_question(prompt)

        context = self._section(prompt, "Context:", "Question:")
        question = self._section(prompt, "Question:", "Answer:")
        source, content = self._first_context_block(context)
        snippet = self._best_sentence(question, content)

        if not snippet:
            return "The retrieved context does not mention the answer."
        return f"{snippet}\n\nSource: Content from [{source}]"

    def _latest_question(self, prompt: str) -> str:
        marker = "Latest question:"
        if marker not in prompt:
            return prompt.strip()
        return prompt.split(marker, maxsplit=1)[1].split("Standalone query:")[0].strip()

    def _section(self, text: str, start: str, end: str) -> str:
        if start not in text:
            return ""
        section = text.split(start, maxsplit=1)[1]
        if end in section:
            section = section.split(end, maxsplit=1)[0]
        return section.strip()

    def _first_context_block(self, context: str) -> tuple[str, str]:
        source_block_pattern = (
            r"Content from \[(?P<source>[^\]]+)\]:\n"
            r"(?P<content>.*?)(?=\n\nContent from \[|\Z)"
        )
        match = re.search(
            source_block_pattern,
            context,
            flags=re.DOTALL,
        )
        if not match:
            return "Unknown Document", context.strip()
        return match.group("source"), match.group("content").strip()

    def _best_sentence(self, question: str, content: str) -> str:
        token_pattern = r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+"
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。])\s+|\n+", content)
            if sentence.strip()
        ]
        if not sentences:
            return ""

        keywords = {
            token.lower()
            for token in re.findall(token_pattern, question)
            if len(token) > 1
        }
        if not keywords:
            return sentences[0]

        def score(sentence: str) -> int:
            lowered = sentence.lower()
            return sum(1 for keyword in keywords if keyword in lowered)

        return max(sentences, key=score)
