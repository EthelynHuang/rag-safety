import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from pipeline import answer_query


def query(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""

    response = answer_query(question)

    sources_md = ""
    for i, s in enumerate(response.sources, 1):
        title = s.title or "Untitled"
        authors = s.authors or []
        author_str = ", ".join(authors) if authors else "Unknown"
        link = f"[{title}]({s.url})" if s.url else title
        sources_md += f"{i}. {link}  \n   Authors: {author_str}\n\n"

    return response.answer, sources_md.strip()


with gr.Blocks(title="AI Safety RAG") as demo:
    gr.Markdown("## AI Safety Research Assistant")
    question = gr.Textbox(label="Question", placeholder="Ask about AI safety or alignment...", lines=2)
    submit = gr.Button("Submit", variant="primary")
    answer_out = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources_out = gr.Markdown(label="Sources")

    submit.click(fn=query, inputs=question, outputs=[answer_out, sources_out])
    question.submit(fn=query, inputs=question, outputs=[answer_out, sources_out])

if __name__ == "__main__":
    demo.launch()
