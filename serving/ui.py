import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def query_api(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    response = requests.post(f"{API_URL}/query", json={"query": question}, timeout=60)
    response.raise_for_status()
    data = response.json()

    answer = data.get("answer", "")

    sources = data.get("sources", [])
    sources_text = ""
    for i, s in enumerate(sources, 1):
        title = s.get("title") or "Untitled"
        url = s.get("url") or ""
        authors = s.get("authors") or []
        author_str = ", ".join(authors) if authors else "Unknown"
        sources_text += f"{i}. {title}\n   Authors: {author_str}\n   URL: {url}\n\n"

    return answer, sources_text.strip()


with gr.Blocks(title="AI Safety RAG") as demo:
    gr.Markdown("## AI Safety Research Assistant")
    question = gr.Textbox(label="Question", placeholder="Ask about AI safety or alignment...", lines=2)
    submit = gr.Button("Submit", variant="primary")
    answer_out = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources_out = gr.Textbox(label="Sources", lines=8, interactive=False)

    submit.click(fn=query_api, inputs=question, outputs=[answer_out, sources_out])
    question.submit(fn=query_api, inputs=question, outputs=[answer_out, sources_out])

if __name__ == "__main__":
    demo.launch()
