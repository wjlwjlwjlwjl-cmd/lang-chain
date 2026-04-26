from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOpenAI(
    model="qwen-turbo"
)

# 字符串模板
prompt_template = PromptTemplate.from_template("translate to {language}")
print(prompt_template.invoke("Chinese"))

prompt_template = PromptTemplate(
    input_variables=["languages"],
    template="translate to {languages}"
)
print(prompt_template.invoke("Chinese"))

# 聊天消息模板
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "translate the content into {language}"),
        ("user", "{text}")
    ]
)

message_template = chat_prompt_template.invoke( # 直接通过模板构造消息
    {
        "language": "English", 
        "text": "此曲只应天上有，人间能得几回闻"
    }
)
model.invoke(message_template).pretty_print()

chain = chat_prompt_template | model
for token in chain.stream({
    "language": "Chinese",
    "text": "This melody should only be found in heaven; how often can it be heard among mortals?"
}):
    print(token.content, end="")

# 消息占位符
messages_template = ChatPromptTemplate(
    {
        ("placeholder", "{msgs}")
    }
)

message_to_pass = [
    HumanMessage("中国首都在哪里？"),
    AIMessage("北京"),
    HumanMessage("法国呢？")
]

model.invoke(messages_template.invoke({
    "msgs": message_to_pass
})).pretty_print()
