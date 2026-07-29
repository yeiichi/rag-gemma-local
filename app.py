import gradio as gr

from rag_gemma_local.rag.service import answer_question


def respond(message, history):
    return answer_question(message, history)


with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=respond,
        title="Gemma Local RAG",
        description=(
            "Ask questions against a local English/Japanese document index. "
            "Answers are constrained to retrieved context."
        ),
        examples=[
            "What are the main topics discussed in these documents?",
            "この文書の重要なポイントを要約してください。",
            "Which sources mention operational risks?",
        ],
        textbox=gr.Textbox(placeholder="Type your question here...", container=False),
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
