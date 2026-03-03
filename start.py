from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage,SystemMessage

# 定义聊天模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0, # 采样度，温度越高，答案越天马行空
    max_tokens=10, # token 是文本的基本单位（NLP），1个token约等于4个字符，对于中文一个汉字约等于1.5~2个token
)

# 定义消息
messages=[
    SystemMessage(content="请补全一段故事，1000个字"),
    HumanMessage(content="一只猫正在__")
]

# 定义解析器
parser=StrOutputParser()

# 定义链和执行链
chain=model|parser
print(chain.invoke(messages))