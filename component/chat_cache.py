from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = ChatOpenAI(
    model = "qwen-turbo"
)

'''
简单粗暴的方法
messages = [
    HumanMessage("我是王家乐"),
    AIMessage("你好王家乐，有什么我可以帮你的吗?"),
    HumanMessage("我是谁?")
]'''

# 使用 
cache = {}
def memory_cache(session_id: str) -> BaseChatMessageHistory:
    if session_id not in cache:
        cache[session_id] = InMemoryChatMessageHistory()
    return cache[session_id]

model_with_memory = RunnableWithMessageHistory(model, memory_cache)

config1 = {"configurable": { "session_id": "1"}}
config2 = {"configurable": { "session_id": "2"}}

model_with_memory.invoke(
    "我叫王家乐",
    config = config1
).pretty_print()

model_with_memory.invoke(
    "我是谁？",
    config = config1
).pretty_print()

print('********************************')

model_with_memory.invoke(
    "我是谁？",
    config = config2
).pretty_print()