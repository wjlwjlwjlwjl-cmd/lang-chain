from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

model = init_chat_model("qwen_turbo", model_provider="DeepSeek", temperature = 1.0)
print(f"qwen-turbo said: {model.invoke('who are you').content}")

# 可配置模型
configurable_model = init_chat_model(temperature = 1.0)
configurable_result = configurable_model.invoke("who are you", config = {"configurable":{"model": "deepseek-chat"}})
print(f"deepseek said{configurable_result.content}")

# 具有默认值的可配置模型
configurable_model_default_arg = init_chat_model(
    model = "deepseek-chat",
    temperature = 1.0,
    configurable_fields = ("model", "temperature", "model_provider"),
    config_prefix = "first"
)
configurable_result_default_arg = configurable_model_default_arg.invoke(
    "who are you",
    config={
        "configurable": {
            "first_model": "deepseek-chat"
        }
    }
)
