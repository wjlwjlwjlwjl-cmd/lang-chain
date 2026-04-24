import time
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def example():
    async def drinking(): # 定义一个协程函数
        print("start drinking")
        await asyncio.sleep(5) # 该协程开始等待
        print("finish drinking")
    async def reply(): 
        print("start reply")
        await asyncio.sleep(2)
        print("finish reply")

    async def main():
        begin = time.time()
        task1 = asyncio.create_task(drinking()) # 创建一个协程
        task2 = asyncio.create_task(reply())
        
        await task1 # 等待该协程完成
        await task2
        end = time.time()
        print(end - begin)

    asyncio.run(main()) # 创建事件循环，并将 main 放到事件循环中执行

model = ChatOpenAI(
    model = "qwen-turbo"
)

def astream_test():
    async def async_output():
        async for chunk in model.astream("讲一个不少于1000字的言情小故事"):
            print(chunk.content, end="", flush=True)
    asyncio.run(async_output())

parser = StrOutputParser()

chain = model | parser
for chunk in chain.stream("写一首关于爱情的诗歌"):
    print(chunk, end = "")