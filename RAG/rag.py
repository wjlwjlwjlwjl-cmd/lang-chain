from langchain_openai import  ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

chat_model = ChatOpenAI(
    model="qwen-turbo"
)
embedding = OllamaEmbeddings(
    model="all-minilm"
)
config=RedisConfig(
    redis_url="redis://localhost:6379",
    index_name="QA",
    metadata_schema=[
        {"name": "category", "type": "tag"},
        {"name": "num", "type": "numeric"},
    ]
)
redis_store = RedisVectorStore(
    embeddings=embedding,
    config=config
)
retriever = redis_store.as_retriever()

@chain
def format_doc(doc: Document) -> str:
    return "\n\n".join(Document.page_content)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", """
        你是context理解大师，请你根据我给你的context，回答我的question，如果context中没有提及，就回答不知道，也不要回答文档之外的任何内容
        context: {context},
        question: {question},
        answer:
        """)
    ]
)
chain = {"context": retriever | format_doc, "question": RunnablePassthrough()} | chat_template | chat_model | StrOutputParser()

for token in chat_model.stream("LangChain的流式传输"):
    print(token.content, end="")