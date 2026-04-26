from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

model = ChatOpenAI(
    model = "qwen-turbo"
)
examples = [
    {"input": "1a1=?", "output": "2"},
    {"input": "1a2=?", "output": "3"},
    {"input": "3a3=?", "output": "6"},
]

chat_template = ChatPromptTemplate(
    [
        ("user", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=chat_template, 
    examples=examples
)

final_prompt = ChatPromptTemplate(
    [
        ("system", "你是一个数学专家"),
        few_shot_template,
        ("human", "{input}")
    ]
)

chain = final_prompt | model
chain.invoke({"input": "15a10等于多少?"}).pretty_print()
