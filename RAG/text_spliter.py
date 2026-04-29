from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

text_loader = TextLoader("/info.md", encoding="utf-8")
doc = text_loader.load()

def length_spliter():
    char_spliter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separator=" ", # 会按照 ["\n\n", "\n", " ", ""]的顺序尝试划分
    )

    for chunk in char_spliter.split_documents(doc):
        print(chunk.page_content)
        print(chunk.metadata)

def tiktoken_spliter():
    char_spliter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=20,
        encoding_name="cl100k_base"
    )
    for chunk in char_spliter.split_documents(doc)[:10]:
        print(chunk.page_content)
        print(chunk.metadata)

def recursive_text_spliter():
    recursive_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,
        chunk_overlap=20,
        encoding_name="cl100k_base",
        separators=["\n\n", "\n", "，", "。", " ", ""]
    )
    for chunk in recursive_spliter.split_documents(doc)[:10]:
        print(chunk.page_content)
        print(chunk.metadata)

embeddings=OpenAIEmbeddings(
    model="text-embedding-v2",  # 魔搭模型ID
    dimensions=768  # 向量维度（与模型匹配）
)
vec=embeddings.embed_query("测试中文文本")
print(f"向量维度: {len(vec)}")  # 输出769
