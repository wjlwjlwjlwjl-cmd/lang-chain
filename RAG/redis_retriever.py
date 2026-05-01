from langchain_redis import RedisVectorStore, RedisConfig
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

embeddings=OllamaEmbeddings(
    model="all-minilm"
)
config=RedisConfig(
    index_name="QA",
    redis_url="redis://localhost:6379",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"}
    ]
)
redis_store = RedisVectorStore(
    embeddings=embeddings,
    config=config
)

def test_as_retriever():
    retriever = redis_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10
        }
    )
    docs = retriever.invoke("LangChain的流式传输")
    for doc in docs:
        print(doc.page_content)
        print('*' * 20)

@chain 
def chain_retriever(input: str) -> List[Document]:
    return redis_store.similarity_search(input, k=1)

def test_chain_retriever():
    ret = chain_retriever.invoke("LangChain的流式传输")
    for doc in [document.page_content for document in ret]:
        print(doc)
        print("*" * 30)

test_chain_retriever()