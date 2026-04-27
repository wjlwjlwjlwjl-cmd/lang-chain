from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import tool_example_to_messages
from typing import List, Optional
from pydantic import Field, BaseModel
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='qwen-turbo'
)

class Person(BaseModel):
    name: Optional[str] = Field(description="姓名")
    skin_color: Optional[str] = Field(description="这个人的肤色")
    hair_color: Optional[str] = Field(description="这个人的发色")
    height_in_meters: Optional[str] = Field(description="这个人以米为单位的身高")

class Data(BaseModel):
    people: List[Person]

examples = [
    (
        "海洋是⼴阔⽽蓝⾊的。它有两万多英尺深。",
        Data(people=[]), # 没有⼈物信息的情况
    ),
    (
        "⼩强从中国远⾏到美国。",
        Data(
            people=[
                Person(name="⼩强", height_in_meters=None, skin_color=None,
                hair_color=None)
            ]
        ), # 部分信息缺失的情况
    )
]

chat_message = ChatPromptTemplate(
    [
        ("system", "你是一个任务信息提取专家，如果给你的描述中，没有相关信息，则相应字段为 None"),
        ("placeholder", "{example_messages}"),
        ("user", "{new_message}")
    ]
)

example_messages = []
for txt, tool_call in examples:
    if tool_call.people:
        ai_response = "未识别到人"
    else:
        ai_response = "识别到人"
    example_messages.extend(
        tool_example_to_messages(
            txt, [tool_call], ai_response = ai_response
        )
    )

model_with_structured_output = model.with_structured_output(Data)
chain = chat_message | model_with_structured_output
print(chain.invoke({"example_messages": example_messages, "new_message": "篮球场上，⾝⾼两⽶的中锋王伟默契地将球传给⼀⽶七的后卫挚友李明，完成⼀记绝杀。" "这对⽼友⽤⼗年配合弥补了⾝⾼的差距。"}))