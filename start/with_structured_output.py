from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import Field, BaseModel
from typing import TypedDict, Optional, List, Union
from typing_extensions import Annotated

class Joke(BaseModel):
    """给用户讲一个笑话 """
    setup: str = Field(description="笑话的开头")
    punchline: str = Field(description="笑话的笑点")
    rating : Optional[int] = Field(default=None, description="笑话的评分(1~10)")

class Jokes(BaseModel):
    """给用户提供的几个笑话"""
    jokes: List[Joke] = Field(description="笑话的合集")


'''class Joke(TypedDict):
    setup = Annotated[str, "笑话的开头"]
    punchline = Annotated[str, "笑话的笑点"]
    rating = Annotated[Optional[int], Field(default=None, description="笑点评分(1~10)")]
'''

'''json_schema = {
    "title": "joke",
    "description": "给⽤⼾讲⼀个笑话。",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "这个笑话的开头",
        },
        "punchline": {
            "type": "string",
            "description": "这个笑话的妙语",
        },
        "rating": {
            "type": "integer",
            "description": "从1到10分，给这个笑话评分",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
'''

class Dialog(TypedDict):
    """对用户查询的内容进行回应"""
    response: Annotated[str, Field(description="对用户的查询进行礼貌的回应")]

class Standard(BaseModel):
    output: Annotated[Union[Dialog, Joke], Field(description="最后输出内容的要求")]

model = ChatOpenAI(
    model="qwen-turbo",
    max_tokens=1024
)
model_structured_output = model.with_structured_output(Standard)

result = model_structured_output.invoke("讲一个关于电脑的笑话")
print(result)
result = model_structured_output.invoke("你好")
print(result)