from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

prompt_template = PromptTemplate(
    template="Input: {input}\nOutput: {output}\n",
    input_variables=["input", "output"]
)

examples = [
    {"input": "See Spot Run", "output": "看见Spot在跑"},
    {"input": "My Dog Barks", "output": "我的狗在叫"},
    {"input": "Spot can Run", "output": "Spot可以跑"},
]

ngram_overlap_example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=prompt_template,
    threshold=0.0
)

few_shot_template = FewShotPromptTemplate(
    example_selector=ngram_overlap_example_selector,
    example_prompt=prompt_template,
    prefix="根据输入进行输出",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

print(few_shot_template.invoke({"input": "Spot can Run Fast"}).to_messages()[0].content)