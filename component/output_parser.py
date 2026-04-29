from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate

model = ChatOpenAI(
    model = "qwen-turbo"
)

def str_output_parser_test():
    output_parser = StrOutputParser()
    chain = model | output_parser
    for token in chain.stream("讲一个关于鼠标的笑话"):
        print(token, end="", flush=True)

def pydantic_output_parser_test():
    class Joke(BaseModel):
        setup: Optional[str] = Field(default="", description="笑话的开头"),
        punchline: Optional[str] = Field(default="", description="笑话的妙语"),
        rate: Optional[int] = Field(default="", description="笑话的评分（1～10）")

    pydantic_output_parser = PydanticOutputParser(pydantic_object=Joke)
    prompt_template = PromptTemplate(
        template="回答用户请求：{format_instruction} \n {query}\n",
        input_variables=["query"],
        partial_variables={"format_instruction": pydantic_output_parser.get_format_instructions()}
    )
    chain=prompt_template | model | pydantic_output_parser
    print(chain.invoke({"query": "讲一个关于电脑的笑话"}))

def json_output_parser_test():
    output_parser = JsonOutputParser()
    prompty_template = PromptTemplate(
        template="answer user's question: {input}\n {format_instruction}",
        input_variables=["input"],
        partial_variables={"format_instruction": output_parser.get_format_instructions()}
    )
    chain=prompty_template|model|output_parser
    print(chain.invoke({"input": "讲一个关于键盘的笑话"}))
    print(chain.invoke({"input": "讲一个关于电脑的笑话"}))
    print(chain.invoke({"input": "讲一个关于鼠标的笑话"}))

def json_output_parser_with_pydantic_test():
    class Joke(BaseModel):
        setup: Optional[str] = Field(default="", description="笑话的开头"),
        punchline: Optional[str] = Field(default="", description="笑话的妙语"),
        rate: Optional[int] = Field(default="", description="笑话的评分（1～10）")
    output_parser = JsonOutputParser(pydantic_object=Joke)
    prompty_template = PromptTemplate(
        template="answer user's question: {input}\n {format_instruction}",
        input_variables=["input"],
        partial_variables={"format_instruction": output_parser.get_format_instructions()}
    )
    chain=prompty_template|model|output_parser
    print(chain.invoke({"input": "讲一个关于键盘的笑话"}))

json_output_parser_test()
json_output_parser_with_pydantic_test()