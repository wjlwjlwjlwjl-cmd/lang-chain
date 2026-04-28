from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_loader = TextLoader("/home/human/Git/mysql/basic.md")
doc = text_loader.load()

char_spliter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=" ", # 会按照 ["\n\n", "\n", " ", ""]的顺序尝试划分
)

for chunk in char_spliter.split_documents(doc):
    print(chunk.page_content)
    print(chunk.metadata)
    print('------------------------------------------')