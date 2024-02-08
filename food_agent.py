from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)

from llm import initialize_openai_gpt4
from tools import image_description

_ = load_dotenv(find_dotenv(), override=True)

# Choose the LLM that will drivfoode the agent
# Only certain models support this


def food_agent():
    llm = initialize_openai_gpt4()

    tools = [image_description]
    # Construct the OpenAI Tools agent

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
# role
diabetes food judger

#goal 
you are a react agent, you are asked to using image description tool multiple times(to collect enough information from multiple generation,according to the return info adjust your query) to calculate the carbonhydrate.
based on the result you need to give a recommendation to the user, and provide the reason for the recommendation.

# workflow
- firstly construct query to describe the food items and size in the image
- according to the description, calculate the carbonhydrates
- return the result with specific structure

# constraints
- return in Chinese
- only return infomation as the structure below, don't provide extra information

# return structure (strict)

整体评价: "超棒","还行", "小心","别吃" \n
营养成分表: \n
an table contains '食物项目','尺寸','数量','碳水含量(g)','蛋白质(g)','脂肪(g)','总卡路里(卡)','GI'(低,中,高)
参考依据：成分表的参考来源
分析:
        """,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return agent_executor


# result = agent_executor.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="http://s7xl013pd.hn-bkt.clouddn.com/3087.jpg_wh300.jpg"
#             )
#         ],
#     }
# )

# print(result["output"])
