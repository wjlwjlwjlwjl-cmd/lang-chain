from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# tool 使用 @tool 和 Python 函数定义
# 函数名、类型提示、文档说明（注释）都会作为参数的一部分传递给工具 schema
# 所谓工具 schema，起到的作用类似与 C++ 类型萃取去检查参数类型，他本身并不是数据结构，但是可以去检查其他具体 Json 的结构、类型是否符合要求
class multiplyInput(BaseModel):
    """this function multiply two number"""
    a: int = Field(..., description="first arg")
    b: int = Field(..., description="second arg")
 
@tool(args_schema = multiplyInput)
def multiply(a, b)-> int:
    return a * b

@tool
def add(
        a: Annotated[int, Field(..., description= "First Arg")],
        b: Annotated[int, Field(..., description="Second Arg")]
)->int:
    """add two integer"""
    return a + b

def fetch_data(url, max_retries = 3):
    """
        fetch data from the given url

        Args:
            url(str): 要获取数据的url
            max_retries(integer, optional): 最大尝试次数，可选
        Returns:
            dicts: 从url 返回的 json 响应
    """
    print("data_fetched")

print(multiply.invoke({"a": 2, "b": 3}))
print(multiply.name)
print(multiply.description)
print(multiply.args)

print(add.invoke({"a": 1, "b": 2}))
