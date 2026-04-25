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

### 8.4 SSE 协议

流式传输需要服务器向客户端主动发送消息。首先我们想到的可能是 WebSocket 协议，这也确实可以，但是需要服务端维护一个长连接，是具有额外的开销。

几乎所有的 LLM 对于流式传输都是用的是 SSE 协议（Server-Sent Event），基于 HTTP 协议，CS 之间建立连接之后，Server 会返回具有 `Content-Type: text/event-stream;charset=utf-8 Connection: keep-alive` 报头的信息，表示当前为流式传输，客户端不要关闭连接。SSE 协议中，客户端后续不能主动给服务器发送消息，对于这种 LLM 的流式传输的场景，本来就不需要客户端发送第一个请求后，后续继续请求，所以 SSE 更符合使用情景

LangChain 的流式传输，并没有自己封装任何协议，是依赖于大模型供应商提供的流式传输的能力，通过 SSE 协议完成的。其中，AIMessageChunk 是 LangChain 根据大模型供应商的 SSE 接口转换而成的
## 九、LangChain 核心组件

### 9.1 消息

* LangChain 中，提供的 SystemMessage、HumanMessage、AIMessage、AIMessageChunk、ToolMessage，都是对不同大模型厂商的接口封装，均继承自 BaseMessage

* BaseMessage 提供了一些通用方法，如 pretty_print，类似 git log --pretty的思想，更清晰美观的展示消息

* LangChain 对话模式

	* 模式一：SystemMessage ---> \[HumanMessage ---> AIMessage\] ---> \[HumanMessage ---> AIMessage\]...... 其中方括号表示一轮对话

	* 模式二：SystemMessage ---> \[HumanMessage ---> AIMessage ---> ToolMessage ---> AIMessage\]

### 9.2 消息缓存

在使用 LLM 时，我们发现他是能记住一定范围内的上下文的。在 LangChain 中，如果只是使用简单的 invoke 或者 stream 可以发现它并没有记忆：其实 LLM 本身是不具备记忆属性的，每一次调用都是一次全新的推理过程，把上面几轮的用户消息和模型的消息给它，就是作为它的“记忆”，组合拼接成含有“记忆”的答案。

比如下面最简单粗暴的方法，就可以实现记忆的功能

```python
messages = [
    HumanMessage("我是张三"),
    AIMessage("你好张三，有什么我可以帮你的吗?"),
    HumanMessage("我是谁?")
]
model.invoke(messages).pretty_print();
```

回答如下，可以发现 LLM 是知道我叫什么的。

```shell
================================== Ai Message ==================================
你是张三！😊 很高兴再次见到你。有什么我可以帮你的吗？或者你想聊些什么？
```

当然这种方法太过简单粗暴了。LangChain 为我们提供了 BaseChatMessageHistory、InMemoryChatMessageHistory 和 RunnableWithMessageHistory，实现上下文的记忆功能。其中，我们使用 invoke 的 config 字段传入 session_id，用来区分不同会话的上下文

```python
cache = {}
def memory_cache(session_id: str) -> BaseChatMessageHistory:
    if session_id not in cache:
        cache[session_id] = InMemoryChatMessageHistory()
    return cache[session_id]
model_with_memory = RunnableWithMessageHistory(model, memory_cache)
config1 = {"configurable": { "session_id": "1"}}
model_with_memory.invoke(
    "我叫王家乐",
    config = config
).pretty_print()
model_with_memory.invoke(
    "我是谁？",
    config = config
).pretty_print()
```

不过这种方法现在已经不建议使用了，到了 LangGraph 持久化再往下说

### 9.3 消息裁剪

一个 LLM 能够处理的上下文是由长度限制的，超过这个限制就需要裁剪，LangChain 允许我们自己设置裁剪策略。

```python
trimmer = trim_messages(
    max_tokens=10, # 最大 token 限制
    token_counter=model, # 通过 语言模型的令牌计数统计
    strategy="last", # 保留最后的消息，如果为 "first" 就是保留最新的消息
    allow_partial=False, # 允许消息从中间被裁剪，一般不允许，会导致内容含义改变
    include_system=True, # 是否总是包含第一条 SystemMessage，建议包含，因为其包含对聊天模型的特殊说明
    start_on='human' # 除了第一条 SystemMessage，保留的第一套消息的类型
)
```

需要注意的是，并不是所有 LLM 都支持通过语言模型的令牌计数来限制 token，比如 qwen-turbo。这里就可以使用另一种方式，通过消息数量来裁剪
```python
trimmer = trim_messages(
    max_tokens=10, # 此时 max_token 表示最大消息数量
    token_counter=len,
    strategy="last",
    allow_partial=False,
    include_system=True,
    start_on='human'
)
```

### 9.4 消息过滤

除了裁剪，有的时候我们想把所有历史中指定的内容交给 LLM，这时候就需要会话历史裁剪。Langchain 在 langchain_core.messages 中提供了 filter_message 来进行过滤，可以通过消息类型、消息 id 等进行过滤，例如：

```python
messages = [
    SystemMessage("you are a good assistant", id='1'),
    HumanMessage("My name is John", id='1'),
    AIMessage("Hello, John", id='2'),
    HumanMessage("I'm very happy today", id='2'),
    AIMessage("Congratulations! Why?", id='3'),
    HumanMessage("Because I hava good grades in my exam! Remember what's my name?", id='3')
]
filtered_message = filter_messages(messages, include_types=[HumanMessage])
model.invoke(filtered_message).pretty_print()
filtered_message  = filter_messages(messages, include_ids='3') # 这里也可以换成 exclude_...
model.invoke(filtered_message).pretty_print()
```

输出内容如下，可以看到，第二次发送其实 LLM 并不知道我们叫什么（瞎猜不算）

```text
================================== Ai Message ==================================
I remember your name is John! 🎉 That's wonderful that you got good grades on your exam. I'm happy for you, John! What are you going to do now that you're feeling so great?
================================== Ai Message ==================================
Congratulations! I'm so happy for you! 🎉 Your name is [Your Name], right? (I might need a reminder if you've told me before!) What's your favorite subject?
```

### 9.5 消息合并

在历史记录中，可能会出现多个同类型消息连在一起的情况（比如多个 HumanMessage、SystemMessage 连在一起），有些 LLM 不允许这种情况，所以我们可以通过 LangChain 的 merge_message_run 来解决这个问题

```python
messages=[
    SystemMessage("you are a coding assistant"),
    SystemMessage("you always feel happy to answer user's questions, but always use Chinese to answer"),
    HumanMessage("Do you think LangChain is good to use?"),
    HumanMessage("or do you think I should use LangGraph"),
    AIMessage("Would you like to tell me, why you wanna using them?"),
    AIMessage("Then I can give you more infomation?"),
    HumanMessage("Give me some introduction about them")
]
merged = merge_message_runs(messages) # 方式一：直接将消息合并
print(model.invoke(messages))

merger = merge_message_runs() # 方式二：构造消息合并器，构建链式调用
chain = merger | model
print(chain.invoke(messages))
```

### 9.5 提示词模板

在一个需要批量提出大量类似请求的情境下，为了保证输出的质量和效率，我们可以使用提示词模板。提示词模板可以让我们把精力放在提示词优化上，只需要让应用把相应的变量传给我们即可

#### 9.5.1 字符串模板

```python
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("translate to {language}") # 直接通过字符串模板初始化
print(prompt_template.invoke("Chinese"))

prompt_template = PromptTemplate( # 指定变量、模板初始化
    input_variables=["languages"],
    template="translate to {languages}"
)
print(prompt_template.invoke("Chinese"))
```

#### 9.5.2 消息模板

消息模板在 LangChain 这种直接与聊天模型交互的场景下最为实用。通过指定消息类型和模板的方式，就可以定义一个消息模板。消息模板还实现了 Runnalbe 接口，可以让我们链式调用

消息模板既可以直接 invoke，实例化出模板消息；也可以在流式调用中，通过指定变量值的方式完成调用

```python
chat_prompt_template = ChatPromptTemplate(
    [
        ("system", "translate the content into {language}"),
        ("user", "{text}")
    ]
)
message_template = chat_prompt_template.invoke(
    {
        "language": "English",
        "text": "此曲只应天上有，人间能得几回闻"
    }
)

chain = chat_prompt_template | model
for token in chain.stream({
    "language": "Chinese",
    "text": "This melody should only be found in heaven; how often can it be heard among mortals?"
}):
```

### 9.6 从 LangChain_Hub 获取提示词

LangChain_Hub 可以认为是 LangChain 的 GitHub，有许多优质的提示词和模板，我们可以直接像 git 一样拉去下来使用，比如下面使用 prompt_maker 的例子

```python
from langsmith import Client
model = ChatOpenAI(
    model="qwen-turbo"
)
client = Client()
prompt = client.pull_prompt("hardkothari/prompt-maker")
while True:
    task = input("请输入你的任务（输入 quit 以退出）：\n")
    if task == 'quit':
        break
    task_prompt = input("请输入你的任务的提示词，后续会自动优化：\n")
    chain = prompt | model
    final_prompt = chain.invoke({
        "task": task,
        "lazy_prompt": task_prompt
    })
    for token in model.stream([final_prompt]):
        print(token.content, end='')
    print()
```