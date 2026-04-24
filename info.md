LangChain 完整学习笔记（纯Markdown精简版）

## 一、LangChain 宏观定义

LangChain 是大模型与实际AI应用之间的桥梁。

- 裸调用API：淌水过河，需手动处理底层细节，开发繁琐。
    
- 使用LangChain：走桥过河，封装通用能力（消息管理、工具调用等），降低开发成本。
    

## 二、大模型核心概念

### 2.1 模型

海量数据中总结规律的程序/函数，传统模型多单任务专精。

### 2.2 大语言模型（LLM）

基于万亿级参数神经网络，以自监督/半监督方式训练的通用语言模型。

- 大规模神经网络：海量参数模拟人脑神经元，动态调整权重习得能力。
    
- 自监督学习：无标准答案，通过文本掩码、上下文预测自学。
    
- 半监督学习：少量标注数据打底，海量无标注数据自我迭代。
    
- 核心能力：基于上下文预测后续文本，输出人类可读语言。
    

### 2.3 大模型定位

大模型=AI大脑（自带通用知识）；LangChain补充：实时信息、外部工具、记忆、结构化输出。

## 三、提示词工程核心

核心原则：严格限定范围、角色、规则、格式，消除模糊输出。

### CO-STAR 原则

Context（上下文）、Objective（目标）、Step（步骤）、Tone（口吻）、Audience（受众）、Response（输出格式）

- 少样本提示：示例+标准答案，复刻逻辑。
    
- 思维链（CoT）：完整推理步骤示例，引导分步思考。
    
- 零样本CoT：结尾加“请一步一步写出思考过程”。
    
- 自我批评迭代：要求AI按标准自查修正。
    

## 四、大模型接入方式

- API-Key调用：轻量化、开箱即用，隐私性弱。
    
- 云端SaaS SDK接入：适配简单业务。
    
- 本地私有化部署（推荐隐私/复杂场景）：突破上下文限制、支持私有数据、管控合规性。
    

## 五、LangChain 核心代码实战（含Python语法解释）

### 5.1 基础消息模型调用

核心消息类型：SystemMessage（系统角色）、HumanMessage（用户请求）、AIMessage（模型响应）

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 初始化模型（Python关键字参数，config配置推荐环境变量，避免硬编码）
model = ChatOpenAI(
    model="qwen-turbo",  # 模型名称（Python字符串参数）
    # temperature：随机性0~2（Python浮点型），越高越发散
)

# 消息列表（Python列表，存储消息对象）
messages = [
    SystemMessage("将英文翻译成通顺中文"),
    HumanMessage("hi!")
]

# 调用+解析（Python对象方法调用）
result = model.invoke(messages)  # invoke方法传入消息列表，获取响应
parser = StrOutputParser()       # 实例化解析器（Python类实例化）
print(parser.invoke(result))     # 解析响应并打印（Python打印函数）
```

### 5.2 LCEL 链式调用

借鉴Linux管道思想，| 运算符重载，串联可运行组件（Python运算符重载语法）

```python
# 链式简写（| 管道符，串联模型与解析器）
chain = model | parser
ret = chain.invoke(messages)  # 链式调用，简化代码

# 等价写法（Python对象构造，两种方式）
from langchain_core.runnables import RunnableSequence
chain1 = RunnableSequence(first=model, last=parser)  # 关键字参数构造
chain2 = model.pipe(parser)  # 实例方法调用
```

### 5.3 统一模型初始化 init_chat_model

统一入口创建多厂商模型，支持运行时动态配置参数（Python字典传参语法）

```python
from langchain.chat_models import init_chat_model

# 初始化可配置模型（Python关键字参数，指定可配置字段）
config_model = init_chat_model(
    model="deepseek-chat",
    temperature=1.0,
    configurable_fields=["model", "temperature"],  # 可动态修改的参数（Python列表）
    config_prefix="llm"
)
# 调用时临时修改参数（Python嵌套字典传参）
res = config_model.invoke(
    "你好",
    config={"configurable": {"llm_temperature": 0.3}}
)
```

## 六、LangChain 工具系统

### 6.1 工具作用

打破大模型知识截止、封闭无法联网的局限，扩展外部能力。

### 6.2 自定义工具开发（含三种Schema参数描述方案，完整代码）

核心三要素：函数+类型注解（Python语法）+标准文档注释，自动生成工具Schema。

#### 方案1：基础@tool装饰器 + Google风格文档注释（简单工具首选）

Python函数注解（指定参数/返回值类型）、文档字符串（说明函数用途与参数）

```python
from langchain_core.tools import tool

# @tool装饰器：将普通Python函数转为LangChain工具
@tool
def multiply(a: int, b: int) -> int:  # 类型注解：a、b为int，返回值为int（Python语法）
    """
    multiply two integers
    Args:
        a: First Integer
        b: Second Integer
    """
    return a * b

# 工具调用（Python字典传参，key对应函数参数名）
print(multiply.invoke({"a": 2, "b": 3}))
print(multiply.name)        # 查看工具名称（工具对象属性）
print(multiply.description) # 查看工具描述（来自文档注释）
print(multiply.args)        # 查看工具参数（自动解析）
```

#### 方案2：Pydantic BaseModel + Field（复杂结构化参数）

Python类继承（继承BaseModel）、Field校验（指定参数描述与约束）

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Pydantic输入模型（Python类继承，定义参数结构）
class multiplyInput(BaseModel):
    """this function multiply two number"""
    # Field：指定参数描述，...表示必填（Pydantic语法）
    a: int = Field(..., description="first arg")
    b: int = Field(..., description="second arg")

# 绑定Schema：通过args_schema参数关联Pydantic模型
@tool(args_schema=multiplyInput)
def multiply(a, b) -> int:  # 无需重复写类型注解，由Schema提供
    return a * b
```

#### 方案3：Annotated + Field（无额外类，轻量化）

Python类型注解扩展（Annotated），直接为参数添加描述（无需额外定义类）

```python
from typing_extensions import Annotated
from langchain_core.tools import tool
from pydantic import Field

# Annotated：为参数添加类型+描述（Python类型注解扩展）
@tool
def add(
    a: Annotated[int, Field(..., description="First Arg")],
    b: Annotated[int, Field(..., description="Second Arg")]
) -> int:
    """add two integer"""
    return a + b
```

### 6.3 工具绑定与调用流程

```python
# 工具绑定（Python列表传入工具，bind_tools方法绑定）
tools = [add, multiply]
model_with_tool = model.bind_tools(tools)

# 构造消息（Python列表存储消息）
msg_list = [HumanMessage("100*20等于多少")]
ai_msg = model_with_tool.invoke(msg_list)

# 遍历执行工具（Python for循环，遍历tool_calls）
for call in ai_msg.tool_calls:
    # 字典映射，根据工具名称获取对应工具（Python字典取值）
    tool = {"multiply": multiply}[call["name"]]
    tool_res = tool.invoke(call)  # 执行工具
    msg_list.append(tool_res)     # 结果加入消息列表

# 最终整合回答
final_res = model_with_tool.invoke(msg_list)
```

## 七、结构化输出（完整代码 + Python语法解释）

通过with_structured_output强制约束输出格式，支持4种方式，适配业务序列化需求。

### 7.1 Pydantic 嵌套模型（最常用，支持嵌套）

```python
from pydantic import BaseModel, Field
from typing import List, Optional  # Python类型提示，List表示列表，Optional表示可选

# 嵌套Pydantic模型（Python类继承，支持嵌套定义）
class Joke(BaseModel):
    """给用户讲一个笑话 """
    setup: str = Field(description="笑话的开头")  # str类型，必填
    punchline: str = Field(description="笑话的笑点")
    # Optional[int]：可选int类型，默认None
    rating: Optional[int] = Field(default=None, description="笑话的评分(1~10)")

class Jokes(BaseModel):
    """给用户提供的几个笑话"""
    # List[Joke]：列表类型，元素为Joke模型（Python类型提示）
    jokes: List[Joke] = Field(description="笑话的合集")

# 绑定结构化输出（传入Pydantic模型，指定输出格式）
model_structured_output = model.with_structured_output(Jokes)
message = [HumanMessage("分别讲一个关于唱歌和跳舞的笑话")]
result = model_structured_output.invoke(message)
print(result)  # 直接返回模型对象，可通过属性取值（Python对象属性访问）
```

### 7.2 TypedDict 结构化（轻量字典类型约束）

```python
from typing_extensions import TypedDict, Annotated
from pydantic import Field

# TypedDict：轻量字典类型约束（Python字典类型提示）
class Joke(TypedDict):
    # Annotated：为字典键添加描述（无需实例化，直接约束类型）
    setup = Annotated[str, "笑话的开头"]
    punchline = Annotated[str, "笑话的笑点"]
    rating = Annotated[Optional[int], Field(default=None, description="笑点评分(1~10)")]

# 绑定结构化输出，include_raw=True返回原始输出
model_structured_output = model.with_structured_output(Joke, include_raw=True)
message = [HumanMessage("讲一个关于跳舞的笑话")]
result = model_structured_output.invoke(message)
```

### 7.3 JSON Schema 直接定义（自定义JSON格式）

```python
# Python字典定义JSON Schema，指定字段类型、描述、必填项
json_schema = {
    "title": "joke",
    "description": "给用户讲一个笑话。",
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
    "required": ["setup", "punchline"],  # 必填字段（Python列表）
}

# 绑定JSON Schema
model_structured_output = model.with_structured_output(json_schema)
message = [HumanMessage("讲一个关于跳舞的笑话")]
result = model_structured_output.invoke(message)
```

### 7.4 Union 联合类型（多格式兼容）

```python
from typing import Union  # Python联合类型，支持多种类型兼容

class Standard(BaseModel):
    # Union[Dialog, Joke]：输出可为Dialog或Joke类型（Python联合类型提示）
    output: Annotated[Union[Dialog, Joke], Field(description="最后输出内容的要求")]
```

## 八、流式传输（Python迭代器/协程/生成器）

### 8.1 基础流式 stream（同步，迭代器）

```python
chunks = []
# for循环遍历迭代器（Python迭代器语法）
for chunk in model.stream("讲一个长笑话"):
    chunks.append(chunk)  # 收集所有块（Python列表append方法）
    print(chunk.content, end="", flush=True)  # 实时打印，flush=True刷新缓冲区
```

### 8.2 异步流式 astream（协程，高并发）

```python
import asyncio  # Python异步模块

# 异步函数（async def定义，Python协程语法）
async def async_output():
    # async for：遍历异步迭代器（Python异步迭代语法）
    async for chunk in model.astream("讲一个言情小故事"):
        print(chunk.content, end="", flush=True)

asyncio.run(async_output())  # 运行协程（Python异步运行方法）
```

### 8.3 自定义流式解析器（生成器yield）

```python
from typing import Iterator  # Python迭代器类型提示

# 自定义解析器（生成器函数，yield关键字生成迭代器，Python生成器语法）
def defined_parser(input: Iterator[str]) -> Iterator[str]:
    buffer = ""
    for chunk in input:
        buffer += chunk
        # 按中文标点切割（Python字符串操作：index找下标、切片）
        while '，' in buffer or '。' in buffer:
            if '，' in buffer:
                stop_index = buffer.index('，')
                yield buffer[:stop_index].strip()  # yield生成每一块内容
                buffer = buffer[stop_index+1:]
            elif '。' in buffer:
                stop_index = buffer.index('。')
                yield buffer[:stop_index].strip()
                buffer = buffer[stop_index+1:]
    yield buffer.strip()  # 生成最后一块内容

# 链式调用自定义解析器
parser = StrOutputParser()
chain = model | parser | defined_parser
for chunk in chain.stream("写一首关于爱情的诗歌"):
    print(chunk)
```

## 九、关键优化建议

- 密钥管理：API_KEY、BASE_URL存入环境变量/.env，禁止硬编码。
    
- 参数调优：temperature=0（严谨场景）；0.7~1.2（创意场景）。
    
- 工具开发：注释清晰，提升调用准确率。
    
- 生产优先：结构化输出+工具调用+本地部署，保障隐私与稳定。