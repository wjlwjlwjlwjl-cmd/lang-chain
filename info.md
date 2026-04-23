## 1 What's LangChain?（宏观）
是连接AI模型与实际AI应用的“桥梁”之一。把AI模型和实际AI应用比作河的两岸，过河可以淌水过河，也可以走桥过河，通过api直接调用就是趟水过河，会“打湿鞋子”，而使用LangChain就是走桥过河，因为LangChain为我们封装了许多细节，是我们的开发得以更加顺畅

## 2. What's 大模型（宏观）
大模型几乎学习了互联网上的所有知识，是一本“百科全书 ”。如果将AI应用比作一个人，那么大模型就是大脑，想要大脑正常运行，还需要眼睛查看实时信息、用手脚感知世界、需要记忆，LangChain就是为大模型增加了这些感官
### 2.1 什么是模型
模型，是一个函数或者程序，能够从海量带有标准答案的训练材料中找出规律，从而完成之后交给的任务。通常一个模型只能够或者说擅长处理一种任务
### 2.2 什么是大语言模型
大语言模型是基于大规模神经网络，通过自监督或者半监督的方式，对海量文本进行训练的语言模型
#### 2.2.1 大规模神经网络
用参数来仿照人脑中神经元的作用，在训练过程中根据结果调整参数的值和链接，这些参数和链接是亿甚至万亿级别的，不断调整，直至能够完成任务
### 2.2.2 自监督
比如想学一门外语，没有老师，只给你一本小说，现在你一个词一个词挡住，根据其他词汇推理出被挡住的词，一直重复直到掌握语法体系，这就是自训练的方式
### 2.2.3 半监督
一句话概括其实就是师傅领进门，修行在个人。有外界基于一定量的带有标准答案的训练材料，语言模型根据这些材料总结出规律，后面自己根据自己找到的没有标准答案的训练材料去修正这些规律，这就是半监督的训练
### 2.2.4 语言模型
根据上下文，推导出下一个词语，类似于中文输入法的联想功能，将AI的出的结果以人类能够读懂的方式展现出来
## 3. 提示词的编写
核心原则就是限定范围，让AI知道你的答案想要什么
### 3.1 CO-STAR原则
(1) Context，指明上下文，例如指明角色、环境等
(2) Objective，指明任务目标
(3) Step，指明工作的步骤
(4) Tone，指明需要以什么样的口吻或者语言风格来组织生成内容
(5) Audience，指明目标对象
(6) Response，指明输出结果的格式
### 3.2 少样本提示
给出已经有答案的实例，让AI根据示例进行推导，可以增加成功率
### 3.3 思维链提示
给AI一个问题和这个问题的思考与解决过程的示例，让AI通过比着葫芦画瓢的方式去解决新的类似的问题，但是如果想要解决复杂问题的话，就需要保证示例的正确和清晰详细
### 3.4 自动推理与零样本链式思考
可以不给予AI示例，但是在请求最后加上一句“请一步一步展示思考过程”，就可以让AI在每次生成前都先进行内部思考，提高答案准确度
### 3.5 自我批评与迭代
在请求的最后，制定标准，要求AI检查自己生成的内容

## 4 接入AI的方式
### 4.1 API key
### 4.2 本地部署
### 4.3 sdk接入
对于涉及隐私数据而且本地条件满足的情况下，建议本地部署；本地部署同时也可以解决以下问题
1. 上下文长度过短
2. 官网模型训练时间是有截止日期的，对于后面出现的文档或者其他内容的处理可能较为乏力
3. 复杂任务处理能力较弱，不能够分步骤一点一点解决复杂任务
4. 难以控制输出格式，也难以控制输出内容的合规性

## 5 代码上手

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parser import StrOutputParser

model = ChatOpenAI(
    model="qwen-turbo", # 模型名称
    # api_key
    # base_url
    # max_tokens 最大token限制
    # temperature 采样度，采样度越高，回答越天马行空
    # max_retries 最大重试次数
    # timeout 超时时间
    # ...
)
messages = [
    SystemMessage("Translate the sentence from English to Chinese"), # 
    HumanMessage("hi!")
]
result = model.invoke(messages)
parser = StrOutputParser()
print(parser.invoke(result))
```
api_key，建议设置进系统的环境变量中。当然可以直接明文放到文件中或者 .env 文件中，但是容易造成密钥泄漏。
这里我使用的是 ubuntu，helix 作为编辑器，在 .profile 中设置 OPENAI_API_KEY 和 OPENAI_BASE_URL 两个环境变量（我使用的是**阿里云百炼**的免费模型，所以需要设置 OPENAI_BASE_URL）

* `SystemMessage`，系统消息，一般作为传给大模型的第一条信息，告诉大模型如何启动接下来的行为
* `HumanMessage`，用户消息，负责传递用户的请求信息
* `invoke`，将消息列表传递给这个方法，进行大模型调用
* `result`，接收返回报文，包括： 
	* `content` 正文内容
	* `additional_kwargs` 与消息关联的有效负载数据，对于AI消息，可能包括大模型调用的工具信息
	* `response_metadata` 响应元数据，如模型版本等
	* `usage_metadata` 消息的使用元数据，如 Token 的消耗情况
* `StrOutputParser`，自动解析出大模型返回内容中最可能为聊天模型返回内容的字符串 

**链式调用**
上面总需要手写 invoke 方法，可以采用下面的方式
```python
chain = model | parser;
ret = chain.invoke(messages);
```

这样 `ret` 就是解析过后聊天模型返回的内容，这里的 `|` 思想上类似于 Linux 中的管道，将前面的结果传给后面作为参数，实际上就是通过两个 Runnable 对象构造一个 RunnableSequence 对象，等价于下面两种写法

```python
chain1 = RunnableSequence(first = model, last = parser)
chain2 = model.pipe(parser)
```

## 5.1 init_chat_model

init_chat_model 返回一个与第二个参数 model_provider 对应的 BaseChatModel 对象，但是这里 model_provider 只能使用支持的，比如 qwen 就不支持
```python
model = init_chat_model("qwen_turbo", model_provider="DeepSeek")
print(f"qwen-turbo said: {model.invoke('who are you')}")
```
### 5.1.1 模型配置

使用 init_chat_model 创建的聊天模型，可以在调用时指定 config，通过 json 的形式给出
```python
configurable_model = init_chat_model(temperature = 1.0)
configurable_result = configurable_model.invoke("who are you", 
	config = {
		"configurable":{
			"model": "deepseek-chat",
			"max_token": 100
		}
	}
)
print(f"deepseek said{configurable_result.content}")
```

### 5.1.2 模型参数修改

在初始化聊天模型时，如果某一项已经初始化，则后续不能直接在 invoke 调用时通过 config 重新指定参数

想要允许后续作出修改，可以在 configurable_fields 字段中指定哪些参数允许后续修改，并指定 configurable_prefix 作为区分秋该参数的前缀，采取类似 protobuf 的方式在 invoke 的配置中加上参数，例如
```python
configurable_model_default_arg = init_chat_model(
    model = "deepseek-chat",
    temperature = 1.0,
    configurable_fields = ("model", "temperature", "model_provider"),
    config_prefix = "first"
)
configurable_result_default_arg = configurable_model_default_arg.invoke(
    "who are you",
    config={
        "configurable": {
            "first_model": "deepseek-chat"
        }
    }
)
```
值得注意的是，这里没有遵循上述规定只是会配置不生效，并不会报错

### 5.2 LangChain 工具

LLM 本质上是一个封闭的模型，所以为了让它能够获得外界新的、其他的知识，可以通过 langchain tool 来帮助扩展 LLM 的边界，进行更加复杂的工作

LangChain的工具由装饰类 @tool 和 Python 函数组成，其中必须包含函数名、类型提示、文档注释，他们会作为参数传递给工具 Schema，解析出的属性用来声明这个工具

> **何为工具 Schema？**
> 所谓工具 schema，起到的作用类似与 C++ 类型萃取去检查参数类型，他本身并不是数据结构，但是可以去检查其他具体 Json 的结构、类型是否符合要求

**基本的工具定义**
```python
from langchain_core.tools import tool
def multiply(a: int, b: int)-> int:
    """
       multiply tow integers

       Args:
           a: First Integer
           b: Second Integer 
    """
    return a * b

print(multiply.invoke({"a": 2, "b": 3}))
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

#### 5.2.1 Google 风格的文档字符串

工具 Schema 需要通过 Google 风格的文档字符串解析信息，如果只是定义简单工具的话。Google 风格的字符串通过 Args Returns 等简介明了的关键字，清晰的解释了函数的情况

**示例**
```python
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
```

#### 5.2.2 通过 pydantic 为 Schema 提供信息

pydantic 是 python 用来作为运行时类型检查的类。一个类继承了 BaseModel 之后，就开启了这个类的类型校验。Field 中可以提供除类型本身种类之外的其他校验规则（类本身的种类要求，就对应了我们前面的类型提示），这里我们通过 description 字段来为 Schema 提供参数信息。

在 tool 的 args_schema 参数中，我们可以传入我们这个类，当 tool 的 Python 函数中没有文档注释提供信息时，可以通过 pydantic 提供的信息来作为文档注释，默认该参数是 None

```python
class multiplyInput(BaseModel):
    """this function multiply two number"""
    a: int = Field(..., description="first arg")
    b: int = Field(..., description="second arg")
 
@tool(args_schema = multiplyInput)
def multiply(a, b)-> int:
    return a * b
```

> 关于 **pydantic** 一词
> 可以认为由两部分组成：
> 1. py，即 python
> 2. pedantic，意思是迂腐的、吹毛求疵的、过分注重细节、严格讲究规则、一丝不苟
> 即 pydantic 负责的工作是 python 的严格的类型检查

#### 5.2.3 通过 Annotated 为 Schema 提供信息

Annotated 可以为变量在不改变其类型信息的基础上，添加其他的说明解释，可以直接与 Field 连用

Annotated 将类型和注释放在一起，不需要在写继承 BaseModel 的类，工具 Schema 会自动识别到参数类型，并把 description 作为参数描述交给大模型

```python
from typing_extensions import Annotated
@tool
def add(
        a: Annotated[int, Field(..., description= "First Arg")],
        b: Annotated[int, Field(..., description="Second Arg")]
)->int:
    """add two integer"""
    return a + b
```

> Annotated，意思是带有注释的

### 5.3 绑定工具

#### 5.3.1 bind_tool
使用 bind_tool\[tool1, tool2, ...\] 的方式将调用它的模型，绑定模型到若干个工具，并返回一个 Runnable 对象，后续可以通过其调用 invoke 方法，完成工具的调用；它提供的参数主要是原始提示词字符串或者是消息列表（ \[HumanMessage(content="")\]

tool_choice 可以指定模型调用，`tool_choice="any"`就是告诉大模型至少调用一个工具，哪怕问题与工具毫不相关
#### 5.3.2 AIMessage
`AIMessage`，是 BaseMessage 之一，负责传递大模型调用工具的接获信息，具体来说，包括
* `content`，消息正文
* `additional_kwargs`，与消息正文相关的其他有效载荷，比如具体调用了哪些个工具（`tool_calls`字段）
* `response_metadata`，相应元数据，包括响应标头、令牌计数、调用模型等等

大模型自己决定是否调用工具以及调用何种工具，问一个毫不相关的问题大模型并不会调用绑定的工具，这也侧面意味着我们的文档注释写的越详细清晰，调用就越准确

`AIMessage` 并不负责产生结果，负责产生结果的是下面的步骤。更明确的，`AIMessage` 负责让 LLM 从我们的 HumanMessage 中解析出调用哪个工具、应该给予工具哪些参数，将他们交给本地方法完成调用之后，才产生了结果

> 这里也是为什么在第一次将 HumanMessage 传给 LLM 时，AIMessage.content 不含内容的原因
#### 5.3.3 ToolMessage
在 `AIMessage` 将调用工具、调用参数等传递回来之后，我们就需要在 **本地** 完成方法的调用。随后会产生 `ToolMessage` 包含工具调用的结果。

最后，将所有调用工具产生的 `ToolMessage` 添加到我们的 message 之中，一起交给 LLM，就完成了整个工具的调用流程

```python
tools = [tool1, tool2]
model_with_tools = model.bind_tools(tools)
message = [
    HumanMessage("100*100等于多少 100+100等于多少")
]
ai_msg = model_with_tools.invoke(message)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": tool1, "multiply": tool2}[tool_call["name"].lower()]
    tool_message = selected_tool.invoke(tool_call)
    message.append(tool_message)
result = model_with_tools.invoke(message)
```

#### 5.3.4 其他工具

langchain 中，并不是所有工具都要我们手动实现，有很多其他工具，langchain 已经帮我们封装好了，比如 tavily，可以让 LLM 获取互联网上最新的知识，不过需要魔法上网

### 5.4 结构化输出

langchain 中允许设置 LLM 的输出格式，通过 `with_structured_output`，它会返回一个 Runnable 对象，通过它就可以获得 LLM 格式化的输出

* `include_raw`，是否将原始 LLM 输出内容返回。默认是 False，设置后，相应分为 raw、parsed、parsed_error 三个部分。
	* `raw`，包含 LLM 原始返回的全部信息，包括格式化的解析结果、调用模型、Token 消耗等等 
	* `parsed`，包含解析后的格式化结果，默认返回的内容就是这里面的信息。解析失败，该部分为 None
	* `parsed_error`，如果出错，这个部分用来存放错误信息
#### 5.4.1 通过 pydantic 对象

支持嵌套格式化，但要注意，这里不要造成循环引用 ❌

```python
class Joke(BaseModel):
    """给用户讲一个笑话 """
    setup: str = Field(description="笑话的开头")
    punchline: str = Field(description="笑话的笑点")
    rating : Optional[int] = Field(default=None, description="笑话的评分(1~10)")
class Jokes(BaseModel):
    """给用户提供的几个笑话"""
    jokes: List[Joke] = Field(description="笑话的合集")
model = ChatOpenAI(
    model="qwen-turbo",
    max_tokens=1024
)
model_structured_output = model.with_structured_output(Jokes)
message=[
    HumanMessage("分别讲一个关于唱歌和跳舞的笑话")
]
result = model_structured_output.invoke(message)
```

输出结果
```python
jokes=[Joke(setup='为什么歌手总是喜欢在浴室唱歌？', punchline='为什么歌手总是喜欢在浴室唱歌？因为那里有最好的回声！', rating=5), Joke(setup='跳舞时最怕什么？', punchline='跳舞时最怕什么？被别人踩到脚，尤其是当你是舞者的时候。', rating=4)]
```

#### 5.4.2 通过 TypedDict

TypedDict 用来检查字典的拼写错误、类型不匹配，可以用来作为 `with_structured_output` 的参数，产生 Runnable 对象

```python
class Joke(TypedDict):
    setup = Annotated[str, "笑话的开头"]
    punchline = Annotated[str, "笑话的笑点"]
    rating = Annotated[Optional[int], Field(default=None, description="笑点评分(1~10)")]
model = ChatOpenAI(
    model="qwen-turbo",
)
model_structured_output = model.with_structured_output(Joke, include_raw=True)
message=[
    HumanMessage("分别讲一个关于跳舞的笑话")
]
result = model_structured_output.invoke(message)
```

#### 5.4.3 通过 json

也可以自己定义 json 串，例如：
```python
json_schema = {
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
model = ChatOpenAI(
    model="qwen-turbo",
)
model_structured_output = model.with_structured_output(json_schema)
message=[
    HumanMessage("分别讲一个关于跳舞的笑话")
]
result = model_structured_output.invoke(message)
```

#### 5.4.4 通过 Union 创建具有联合类型属性的父模式

```python
class Standard(BaseModel):
    output: Annotated[Union[Dialog, Joke], Field(description="最后输出内容的要求")]
```