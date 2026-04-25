from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, merge_message_runs
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen-turbo"
)
messages=[
    SystemMessage("you are a coding assistant"),
    SystemMessage("you always feel happy to answer user's questions, but always use Chinese to answer"),
    HumanMessage("Do you think LangChain is good to use?"),
    HumanMessage("or do you think I should use LangGraph"),
    AIMessage("Would you like to tell me, why you wanna using them?"),
    AIMessage("Then I can give you more infomation?"),
    HumanMessage("Give me some introduction about them")
]
merged = merge_message_runs(messages)
print(model.invoke(messages))

print('\n\n\n')

merger = merge_message_runs()
chain = merger | model
print(chain.invoke(messages))