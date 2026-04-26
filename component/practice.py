from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model="qwen-turbo"
)

chat_template = ChatPromptTemplate(
    [
        ("system", "你是一个翻译家"),
        ("user", "将下列内容翻译成{language}, {content}"),
    ]
)

final_message = chat_template.invoke(
    {
        "language": "英文", 
        "content": "明月几时有，把酒问青天"
    }
).to_messages()

parser = StrOutputParser()
chain = model | parser
print(chain.invoke(final_message))

