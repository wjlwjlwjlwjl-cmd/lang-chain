from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, filter_messages
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen-turbo"
)

messages = [
    SystemMessage("you are a good assistant", id='1'),
    HumanMessage("My name is John", id='1'),
    AIMessage("Hello, John", id='2'),
    HumanMessage("I'm very happy today", id='2'),
    AIMessage("Congratulations! Why?", id='3'),
    HumanMessage("Because I hava good grades in my exam! Remember what's my name?", id='3')
]

filtered_message = filter_messages(messages, include_types=[HumanMessage])
model.invoke(filtered_message).pretty_print()

filtered_message  = filter_messages(messages, include_ids='3')
model.invoke(filtered_message).pretty_print()