from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

def PyPDFLoaderTest():
    pdf_loader = PyPDFLoader("/home/human/文档/bite/MySQL/test.pdf")
    doc = pdf_loader.load()
    # noinspection PyCompatibility
    print(f"pdf has {len(doc)} pages")
    print(f"the first page's first 200 words: {doc[0].page_content[:200]}")
    print(f"the first page's metadata {doc[0].metadata}")

def UnstructuredMarkdownLoaderTest():
    markdown_loader = UnstructuredMarkdownLoader("/home/human/Git/mysql/basic.md", mode="elements")
    doc = markdown_loader.load()
    assert len(doc) == 1
    assert isinstance(doc, list)
    print(doc[0].page_content)
    print(doc[0].metadata)
    print(set(document.metadata["catagory"] for document in doc )) # 从右向左执行

UnstructuredMarkdownLoaderTest()