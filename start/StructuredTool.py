from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model = "qwen-turbo"
)

class AddInput(BaseModel):
    a: Annotated[int, Field(..., description="First Arg")]
    b: Annotated[int, Field(..., description="Second Arg")]

def add(a, b):
    return a + b

tool1 = StructuredTool.from_function(
    func = add, 
    description = "add two number together",
    args_schema = AddInput,
    name = "add"
)

class MultiplyInput(BaseModel):
    a: Annotated[int, Field(..., description="First Arg")]
    b: Annotated[int, Field(..., description="Second Arg")]

def multiply(a, b):
    return a * b;

tool2 = StructuredTool.from_function(
    func = multiply,
    description = "multiply two number", 
    args_schema = MultiplyInput,
    name = "multiply"
)

tool = [tool1, tool2]
model_with_tools = model.bind_tools(tool)

ai_msg = model_with_tools.invoke("2 * 6 等于多少")
print(ai_msg.additional_kwargs)
print(ai_msg.tool_calls)
print(ai_msg.content)

print()
print(ai_msg)