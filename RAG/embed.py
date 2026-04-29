from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_loader = TextLoader("../info.md", encoding="utf-8")
doc = text_loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", "。", "，", ""]
)

# 定义嵌入模型
embeddings = OllamaEmbeddings(
    model="all-minilm"
)
texts = [chunk.page_content for chunk in spliter.split_documents(doc)]

# 嵌入文档列表
vectors = embeddings.embed_documents(texts)
print(f"文档的个数为{len(doc)}，为每个文档生成了{len(vectors)}个向量列表")
print(f"第一个文档共有{len(vectors[0])}个向量")
print(f"第二个文档共有{len(vectors[1])}个向量")

# 嵌入单个查询
query_result = embeddings.embed_query("MMR")
print(f"向量维度{len(query_result)}")
print(f"向量的前五个维度{query_result[:5]}")