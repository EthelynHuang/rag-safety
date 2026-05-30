from fastapi import FastAPI
from models import QueryRequest, QueryResponse
from pipeline import answer_query

app = FastAPI()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return answer_query(request.query)
