from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embedding = OllamaEmbeddings(
    model="all-minilm"
)
loader = TextLoader("../info.md", encoding="utf-8")
doc = loader.load()
spliter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
vector_store = InMemoryVectorStore(embedding=embedding)

texts = [document for document in spliter.split_documents(doc)]
ids = vector_store.add_documents(texts) # 增
print(f"共添加了{len(ids)}个文档")
print(f"前三个文档索引为{ids[:3]}")

vecs = vector_store.get_by_ids(ids[:3]) # 查，注意 get_by_ids 必须要传入列表
print(f"前三个文档分片内容为：{[docs.page_content for docs in vecs]}")

vector_store.delete(ids=ids[:1])
vecs = vector_store.get_by_ids(ids[:3])
print(f"删除后，尝试获取之前的文档内容{[docs.page_content for docs in vecs]}")

# 语义向量检索
def _vector_filter(candidate: Document) -> bool:
    return candidate.metadata.get("source") != "xxx"

ret = vector_store.similarity_search(
    query="Runnable",
    k = 2,
    filter=_vector_filter
)
print("*****************************")
print("内容检索结果")
for doc in ret:
    print(doc.page_content, end="")