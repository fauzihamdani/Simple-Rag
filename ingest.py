from pathlib import Path
# from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import litellm
import json
import requests 

load_dotenv(override=True)

MODEL = "ollama/qwen2.5:3b"

DB_NAME = str(Path(__file__).parent.parent / "learn-RAG/preprocessed_db")
collection_name = "docs"
EMBED_MODEL = "ollama/nomic-embed-text"
KNOWLEGDE_BASE_PATH = Path(__file__).parent.parent / "learn-RAG/knowledge-base"
AVERAGE_CHUNK_SIZE = 100
wait = wait_exponential(multiplier=1, min=10, max = 240)

WORKERS = 1


class Result(BaseModel):
    page_content: str
    metadata:dict

class Chunk(BaseModel):
    headline : str = Field(
        description = "A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text : str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content = self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )

class Chunks(BaseModel):
    chunks : list[Chunk]

def save_chunks(chunks, path="chunks_backup.json"):
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Chunks saved to {path}")

def load_chunks(path="chunks_backup.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return [Result(page_content=d["page_content"], metadata=d["metadata"]) for d in data]

def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
     
    documents = []

    for folder in KNOWLEGDE_BASE_PATH.iterdir():
         doc_type = folder.name
         for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source":file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents

def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the shared drive of a company called Insurellm.
    The document is of type: {document["type"]}
    The document has been retrieved from: {document["source"]}

    A chatbot will use these chunks to answer questions about the company.
    You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
    This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document["text"]}

    Respond with the chunks.
    """

def make_messages(document):
    return [
        {"role":"user", "content": make_prompt(document)}
    ]

@retry(wait=wait)
def proccess_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    """
        Create chunks using a number of workers in parallel.
        If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(proccess_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks

def get_embeddings(text):
    response = requests.post(

        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text", "input": text}
    )
    return response.json()["embeddings"][0]

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    texts = [chunk.page_content for chunk in chunks]

    vectors = []
    for text in tqdm (texts, desc = "Embedding"):
        vectors.append(get_embeddings(text))

    collection = chroma.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

if __name__ == "__main__":
    CHUNKS_BACKUP = "chunks_backup.json"
    
    if Path(CHUNKS_BACKUP).exists():
        print("Loading chunks from backup file...")
        chunks = load_chunks(CHUNKS_BACKUP)
    else:
        documents = fetch_documents()
        chunks = create_chunks(documents)
        save_chunks(chunks)  # simpan dulu sebelum embedding
    
    create_embeddings(chunks)
    print("Ingestion complete")

    
    
