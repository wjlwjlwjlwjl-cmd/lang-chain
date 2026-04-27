from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_chroma import Chroma

model = ChatOpenAI(
    model="qwen-turbo"
)

prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="input:{input}, output:{output}"
)

examples=[
    {"input": "happy", "output": "sad"},
    {"input": "sunny", "output": "rainy"}
]

mmr = MaxMarginalRelevanceExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k = 2
)

few_shot_template = FewShotPromptTemplate(
    example_selector=mmr,
    example_prompt=prompt_template,
    prefix="give the adjective of the input word",
    suffix="input: {input}, output:",
    input_variables=["input"]
)

model.invoke(few_shot_template.invoke({"high"})).pretty_print()