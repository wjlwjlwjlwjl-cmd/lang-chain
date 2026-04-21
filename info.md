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