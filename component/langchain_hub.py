from langchain_openai import ChatOpenAI
from langsmith import Client

model = ChatOpenAI(
    model="qwen-turbo"
)

client = Client()
prompt = client.pull_prompt("hardkothari/prompt-maker")

while True:
    task = input("请输入你的任务（输入 quit 以退出）：\n")
    if task == 'quit':
        break
    task_prompt = input("请输入你的任务的提示词，后续会自动优化：\n")

    chain = prompt | model
    final_prompt = chain.invoke({
        "task": task,
        "lazy_prompt": task_prompt
    })
    for token in model.stream([final_prompt]):
        print(token.content, end='')
    print()