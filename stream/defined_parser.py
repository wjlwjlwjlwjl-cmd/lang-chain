from langchain_openai import ChatOpenAI
from typing import Iterator, List
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model = "qwen-turbo"
)

def defined_parser(input: Iterator[str]) -> Iterator[str]:
    buffer = ""
    for chunk in input:
        buffer += chunk
        while '，' in buffer or '。' in buffer:
            if '，' in buffer:
                stop_index = buffer.index('，')
                yield buffer[:stop_index].strip()
                buffer = buffer[stop_index + 1:]
            elif '。' in buffer:
                stop_index = buffer.index('。')
                yield buffer[:stop_index].strip()
                buffer = buffer[stop_index + 1:]
    yield buffer.strip()

parser = StrOutputParser()
chain = model | parser | defined_parser
for chunk in chain.stream("写一首关于爱情的诗歌"):
    print(chunk)