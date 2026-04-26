from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = "qwen-turbo"
)

examples = [
    {
        "question": "李⽩和杜甫，谁更⻓寿？",
        "answer": """
        是否需要后续问题：是的。
        后续问题：李⽩享年多少岁？
        中间答案：李⽩享年61岁。
        后续问题：杜甫享年多少岁？
        中间答案：杜甫享年58岁。
        所以最终答案是：李⽩
        """
    },
    {
        "question": "腾讯的创始⼈什么时候出⽣？",
        "answer": """
        是否需要后续问题：是的。
        后续问题：腾讯的创始⼈是谁？
        中间答案：腾讯由⻢化腾创⽴。
        后续问题：⻢化腾什么时候出⽣？
        中间答案：⻢化腾出⽣于1971年10⽉29⽇。
        所以最终答案是：1971年10⽉29⽇
        """,
    },
    {
        "question": "孙中⼭的外祖⽗是谁？",
        "answer": """
        是否需要后续问题：是的。
        后续问题：孙中⼭的⺟亲是谁？
        中间答案：孙中⼭的⺟亲是杨太夫⼈。
        37后续问题：杨太夫⼈的⽗亲是谁？
        中间答案：杨太夫⼈的⽗亲是杨胜辉。
        所以最终答案是：杨胜辉
        """,
    }
]

chat_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{question}"),
        ("ai", "{answer}")
    ]
)

few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=chat_template,
    examples=examples, 
)

final_message = ChatPromptTemplate(
    [
        few_shot_template,
        ("user", "《星球大战》的导演和《教父》的导演来自同一个国家吗?")
    ]
)

chain = final_message | model
chain.invoke({}).pretty_print()
