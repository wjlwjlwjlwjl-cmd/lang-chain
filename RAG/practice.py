from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader("../info.md", encoding="utf-8")
doc = text_loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
embedding = OllamaEmbeddings(
    model="all-minilm"
)
texts = [document.page_content for document in spliter.split_documents(doc)]
vectors = embedding.embed_documents(texts)
print(f"共生成了{len(vectors)}个向量")
print(f"单个向量维数为{len(vectors[0])}")
print(f"前五个向量为{vectors[:5]}")