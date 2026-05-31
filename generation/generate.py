import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert research assistant answering questions on AI safety. "
    "Answer using only the provided sources. Cite every claim with the source number "
    "(e.g., [1], [2]). If the sources don't contain enough information, respond with: "
    "\"I don't have enough information in the provided sources to answer this question.\""
)


def generate_answer(query: str, chunks: list[dict]) -> str:
    sources_text = ""
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "")
        url = chunk.get("url", "")
        text = chunk.get("text", "")
        sources_text += f"Source {i}:\nTitle: {title}\nURL: {url}\nText: {text}\n\n"

    user_message = f"Here are the sources:\n\n{sources_text}Question: {query}"

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text
