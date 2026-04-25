from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages

model = ChatOpenAI(
    model = "qwen-turbo"
)

messages = [
    SystemMessage("you are a good assistant"),
    HumanMessage("My name is John"),
    AIMessage("Hello, John"),
    HumanMessage("I'm very happy today"),
    AIMessage("Congratulations! Why?"),
    HumanMessage("Because I hava good grades in my exam!")
]

trimmer = trim_messages(
    max_tokens=10,
    token_counter=len,
    strategy="last",
    allow_partial=False,
    include_system=True,
    start_on='human'
)

chain = trimmer | model
result = chain.invoke(messages)
print(result)