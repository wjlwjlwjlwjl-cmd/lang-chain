from langchain_redis import RedisVectorStore, RedisConfig
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

embedding = OllamaEmbeddings(
    model="all-minilm"
)
config = RedisConfig(
    index_name = "QA",
    redis_url = "redis://localhost:6379",
    metadata_schema=[ # 额外每个索引还需要有哪些内容
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"}
    ]
)
redis_store = RedisVectorStore(
    embeddings=embedding,
    config=config
)

spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
docs = TextLoader("../info.md", encoding="utf-8").load()
splited_docs = spliter.split_documents(docs)

for i, document in enumerate(splited_docs, start=1):
    document.metadata["category"] = "QA"
    document.metadata["num"] = i
ids = redis_store.add_documents(splited_docs)

"""
print(f"共生成了{len(ids)}个索引")
print(f"前三个文档的索引是{ids[:3]}")
print('*' * 30)

ids=['QA:01KQH55F8S7AEZZG4S4QGKDT4J', 'QA:01KQH55F8S7AEZZG4S4QGKDT4K', 'QA:01KQH55F8S7AEZZG4S4QGKDT4M']
print('*' * 30)

redis_store.delete(ids[:3])

similarity_ret = redis_store.similarity_search(
    query="Pydantic", 
    k = 2
)
"""