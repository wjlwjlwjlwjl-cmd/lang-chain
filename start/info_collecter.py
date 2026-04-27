from langchain_openai import ChatOpenAI
from typing_extensions import Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, BaseModel
from typing import Optional

model = ChatOpenAI(
    model="qwen-turbo"
)
class Info(BaseModel):
    """人员信息提取"""
    name: Annotated[str, Field("人员的姓名")]
    gender: Annotated[Optional[str], Field(default=None, description="人员的性别")]
    address: Annotated[Optional[str], Field(default=None, description="人员的国籍")]
    outlook: Annotated[Optional[str], Field(default=None, description="人员外貌特征")]

message = [
    SystemMessage("你是人员信息提取大师，能够根据输入提取出人员特征，如果没有相应信息请返回 null"),
    HumanMessage("我叫做王家乐，黑头发，中国人，男生")
]

model_with_structured_output = model.with_structured_output(Info)
result = model_with_structured_output.invoke(message)
print(result)