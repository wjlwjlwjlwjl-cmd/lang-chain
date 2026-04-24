#首先绑定工具，然后添加结构化输出
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
from langchain_core.tools import tool

class search_result(BaseModel):
    """结构化搜索请求"""
    query: str = Field(description="搜索请求")
    result: str = Field(description="搜索结果")

@tool
def web_search(query: str) -> str:
    """
        联网搜索用户请求
        Args:
            query(str): 搜索请求
    """
    return "Ubuntu 最新版本是26.04"

messages = [
    HumanMessage("Ubuntu 最新版本是什么")
]
model = ChatOpenAI(
    model = "qwen-turbo"
)
model_with_tool = model.bind_tools([web_search], tool_choice="any")
ai_msg = model_with_tool.invoke(messages);
for tool_call in ai_msg.tool_calls:
    tool_message = web_search.invoke(tool_call)
    messages.append(tool_message)
model_with_structured_output = model_with_tool.with_structured_output(search_result)
result = model_with_structured_output.stream(messages)
print(result)