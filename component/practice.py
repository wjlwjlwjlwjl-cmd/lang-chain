from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

model = ChatOpenAI(
    model = "qwen-turbo"
)

chat_template = ChatPromptTemplate(
    [
        ("user", "{question}"),
        ("ai", "{answer}")
    ]
)

examples = [
    {"question": "1@2=?", "answer": "3"},
    {"question": "2@2=?", "answer": "4"},
    {"question": "3@2=?", "answer": "5"},
]

few_shot_template = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=chat_template
)

final_messages = ChatPromptTemplate(
    [
        ("system", "你是一个数学大师"),
        few_shot_template,
        ("user", "8@3=?")
    ]
)

model.invoke(final_messages.invoke({})).pretty_print()