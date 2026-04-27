from langchain_core import example_selectors
from langchain_core.example_selectors import LengthBasedExampleSelector, SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_chroma import Chroma

examples=[
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

prompt_template = PromptTemplate(
    input_variables=["input",  "output"],
    template="input: {input}\noutput: {output}"
)

# 通过片段数量选择
length_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=prompt_template, 
    max_length=10 # 按空格、换行、制表符分割之后的片段的数量
)

# 通过语义选择
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(model="text-embedding-v3", chunk_size=1), Chroma, 2)

few_shot_template = FewShotPromptTemplate(
    example_selector=semantic_selector,    
    example_prompt=prompt_template,
    prefix="给出每个输入的反义词",
    suffix="input: {adjective}\n",
    input_variables=["adjective"]
)

print(few_shot_template.invoke({"adjective": "vim"}).to_messages()[0].content)