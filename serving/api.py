import gradio as gr
from fastapi import FastAPI
from models import QueryRequest, QueryResponse
from pipeline import answer_query
from serving.ui import demo

app = FastAPI()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return answer_query(request.query)


app = gr.mount_gradio_app(app, demo, path="/")
