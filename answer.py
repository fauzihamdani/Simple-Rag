from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
import requests


load_dotenv(override=True)


MODEL = "ollama/qwen2.5:3b"
# MODEL = "groq/openai/gpt-oss-120b"
DB_NAME = str(Path(__file__).parent.parent / "learn-RAG/preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "learn-RAG/knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "learn-RAG/summaries"

collection_name = "docs"
embedding_model = "nomic-embed-text"
wait = wait_exponential(multiplier=1, min=10, max=240)

# openai = OpenAI()

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You are a helpful assistant for Insurellm.
IMPORTANT: You MUST answer ONLY based on the context provided below.
look at the information!! look again and again. If the answer is in the context, use it. Do NOT say you don't know if the information exists in the context.
Do NOT use your own knowledge.

Context:
{context}

Answer the user's question based strictly on the context above.
"""


class Result(BaseModel):

    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder, temperature=0)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
        You are in a conversation with a user, answering questions about the company Insurellm.
        You are about to look up information in a Knowledge Base to answer the user's question.

        This is the history of your conversation so far with the user:
        {history}

        And this is the user's current question:
        {question}

        Respond only with a short, refined question that you will use to search the Knowledge Base.
        It should be a VERY short specific question most likely to surface content.
        Use terms likely found in HR documents such as: compensation, salary history, base salary, performance bonus.
        IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
    """
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}], temperature=0)
    # print('response => ',response,"\n")
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model":"nomic-embed-text" ,"input":text}
    )
    return response.json()["embeddings"][0]


def fetch_context_unranked(question):
   
    query = get_embedding(question)

    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    # print(f"Rewritten: {rewritten_question}")
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    # print(f"Top 3 chunks:")
    # for i, chunk in enumerate(reranked[:3]):
    #     print(f"Chunk {i+1}: {chunk.page_content[:200]}")
    return reranked[:FINAL_K]

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages, temperature=0.3)
    return response.choices[0].message.content, chunks



