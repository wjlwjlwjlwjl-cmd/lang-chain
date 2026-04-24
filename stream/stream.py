from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = "qwen-turbo"
)

chunks = []
for chunk in model.stream("讲一个字数不少于1000字的笑话"):
    chunks.append(chunk)
    print(chunk.content, end="", flush=True)
for unit in chunks:
    print(unit)