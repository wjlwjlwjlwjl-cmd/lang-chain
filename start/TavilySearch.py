from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import os
import requests

# 百度千帆搜索函数
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询内容")

def baidu_search(query: str):
    api_key = os.getenv("BAIDU_API_KEY")
    url = "https://qianfan.baidubce.com/v2/ai/search"  # ✅ 正确接口！
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "query": query,
        "top_n": 4   # ✅ 正确参数名
    }

    resp = requests.post(url, headers=headers, json=data)
    result = resp.json()

    # ✅ 正确返回格式！
    if "answer" in result:
        return result["answer"]
    return str(result)

# 包装成 LangChain 标准工具
tool = StructuredTool.from_function(
    func=baidu_search,
    args_schema=SearchInput,
    name="baidu_search",
    description="联网搜索实时信息、天气、新闻、知识"
)

# ===================== 主逻辑 =====================
model = ChatOpenAI(model="qwen-turbo")
model_with_tool = model.bind_tools([tool])

messages = [HumanMessage("今天上海天气？")]

# 第一次调用：AI 决定调用工具
ai_msg = model_with_tool.invoke(messages)

# 执行搜索
for tool_call in ai_msg.tool_calls:
    tool_msg = tool.invoke(tool_call)
    messages.append(tool_msg)

# 第二次调用：AI 生成回答
result = model_with_tool.invoke(messages)
print(result.content)