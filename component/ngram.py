from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}\n"
)

examples = [
    {"input": "See spot run", "output": "看见Spot跑"},
    {"input": "My dog barks.", "output": "我的狗叫。"}, # 完全不相关的实例，会被ngram排除掉
    {"input": "Spot can run.", "output": "Spot可以跑。"},
]

ngram_overlap_example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=prompt_template,
    threshold=-1.0 # threshold 是过滤的门槛，-1.0 表示保留全部只做排序，0.0 表示只删掉完全不重叠的，1.0理论上只保留完全匹配的，>1.0 全部删除
)

few_shot_template = FewShotPromptTemplate(
    example_selector=ngram_overlap_example_selector,
    example_prompt=prompt_template,
    prefix="给出每个输入的中文翻译",
    suffix="Input:{input}\nOutput:",
    input_variables=["input"]
)

print(few_shot_template.invoke({"input":"Spot can run fast"}).to_messages()[0].content)