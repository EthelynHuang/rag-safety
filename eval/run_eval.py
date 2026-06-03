"""
Evaluate the hybrid RAG pipeline with RAGAS.

Run from project root:
    python3 eval/run_eval.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retrieval.search import hybrid_search
from generation.generate import generate_answer
from eval.questions import questions as EVAL_QUESTIONS

CLAUDE_MODEL = "claude-sonnet-4-20250514"


def main():
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(EVAL_QUESTIONS, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")

        chunks = hybrid_search(question, top_k=10)
        answer = generate_answer(question, chunks)
        contexts = [c["text"] for c in chunks if "text" in c]

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(ground_truth)

    dataset = Dataset.from_dict(rows)

    llm = ChatAnthropic(model=CLAUDE_MODEL, temperature=0)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )

    print("\n=== RAGAS scores ===")
    print(dict(result))

    df = result.to_pandas()
    out_path = os.path.join(os.path.dirname(__file__), "results_hybrid.csv")
    df.to_csv(out_path, index=False)
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
