from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model="qwen-turbo",
)
messages = [
    SystemMessage("Translate the sentence from English to Chinese"),
    HumanMessage("who are you!")
]
parser = StrOutputParser()
#chain = model | parser
#chain = RunnableSequence(first = model, last = parser)
model.invoke(messages).pretty_print()
