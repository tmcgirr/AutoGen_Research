import autogen
from autogen import AssistantAgent, UserProxyAgent, Cache, register_function
from typing_extensions import Annotated
import os, json, asyncio

# Tools
from Tools.web_voyager_tool import execute_query

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}], 
}

# Setup for WebVoyagerAssistant
web_voyager_assistant = AssistantAgent(
    name="Web_Voyager_Assistant",
    system_message="Please execute the following web navigation tasks. Store the information in a file if the information is useful. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Setup for Analyst
analyst = AssistantAgent(
    name="Intelligence_Analyst",
    system_message="Please analyze the following data to determine the quality of the information. If the data is not useful, try a different query. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Setup for UserProxyAgent
user_proxy = UserProxyAgent(
    name="User",
    system_message="A proxy for the user.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=10,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

############
# Define and register the function Method 
# register_function(
#     run_web_voyager_assistant,
#     caller=web_voyager_assistant,
#     executor=user_proxy,
#     name="run_web_voyager_assistant",
#     description="Execute the following web navigation tasks.",
# )

@web_voyager_assistant.register_for_execution()
@web_voyager_assistant.register_for_llm(name="Web_Voyager_Assistant", description="Perform web search and navigation.")
async def run_web_voyager_assistant(query: Annotated[str, "Query to execute."]) -> str:
    return await execute_query(query)

@user_proxy.register_for_execution()
@user_proxy.register_for_llm(name="User", description="A proxy for the user.")
def terminate_group_chat(message: Annotated[str, "Message to be sent to the group chat."]) -> str:
    return f"[GROUPCHAT_TERMINATE] {message}"

############
# Setup the group chat with the existing agents
groupchat = autogen.GroupChat(agents=[user_proxy, web_voyager_assistant, analyst], messages=[], max_round=12)

# Configure the GroupChatManager (Revise to remove functions and tools from the config)
llm_config_manager = llm_config.copy()
# llm_config_manager.pop("functions", None)
# llm_config_manager.pop("tools", None)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_manager,
    is_termination_msg=lambda x: "GROUPCHAT_TERMINATE" in x.get("content", ""),
)

# Initialize and start the group chat
message = """
Please find the the commute time from San Diego to Los Angeles by car.
"""

async def initiate_chat():
    with Cache.disk() as cache:
        await user_proxy.a_initiate_chat(
            manager,
            message=message,
            cache=cache,
        )

# Run the chat
asyncio.run(initiate_chat())