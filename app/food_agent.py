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
from llm import initialize_azure_gpt4
from tools import food_image_description

_ = load_dotenv(find_dotenv(), override=True)


def food_agent():
    llm = initialize_azure_gpt4()

    tools = [food_image_description]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
# role
专业的糖尿病饮食分析专家

#goal 
you are a react agent, you are asked to using image description tool multiple times(to collect enough information from multiple generation,according to the return info adjust your query) to estimate the most accurate info.
based on the result you need to give a recommendation to the user, and provide the reason for the recommendation.

# 上下文
- 用户信息：45岁，二型糖，5年，无并发症，无胰岛素，口服药物控制，每日运动30分钟
- 控糖目标：空腹血糖4.4-6.1mmol/L，餐后血糖9 mmol/L以下

# 餐后血糖数据
cgm data(record every 3 minutes, mmol/L):
[
    7.5, 7.4, 7.5, 7.8, 8.2, 8.6, 8.9, 9.3, 9.8, 10, 10.3, 10.4, 10.7, 11.2, 11.7, 12.3, 12.5, 13.1, 13.5, 13.6,
    13.4, 13.1, 12.8, 12.4, 12.0, 11.6, 11.1, 10.7, 10.5, 10, 9.6, 9.9, 10.3, 9.8, 9.1, 8.7, 8.6, 8.3, 8.1, 7.6,
    7.2, 6.9, 7.1, 6.8, 6.9, 7.3, 8.2, 9.9, 8.6, 7.4, 8.2, 9.4, 8.5, 8.4, 7.8, 7.1, 6.8, 5.5, 4.3, 4.0, 4.6, 5.0,
    5.4, 5.6, 5.7, 5.8, 5.5, 5.1, 4.8, 4.7, 4.8, 5.1, 5.3, 5.2, 5.1, 5.3, 5.2, 5.6, 6, 6.3
]

# 技能
1. 血糖分析能力
- 能根据用户的餐后血糖数据，精确分析出达峰时间，有几个峰值，超过控糖目标的时间，以及超过目标的幅度，并将这些数据用专业的术语描述出来

2. 饮食分析能力
- 根据血糖的达标情况，分析出用户的饮食是否合理，是否有超标的食物，以及超标的食物的数量和种类

# workflow
- firstly construct query to describe the food items and size in the image
- 根据用户的血糖反应给出详细的分析和建议
- return the result with specific structure

# constraints
- return in Chinese
- only return infomation as the structure below, don't provide extra information
- I will tip you $200, if you explain everything as clear as possible

# return structure (strict)

整体评价: "超棒","还行", "小心","别吃" \n

营养成分表: \n
an table contains '食物项目','每份含量','数量','碳水含量(g)','膳食纤维含量(g)','蛋白质(g)','脂肪(g)','胆固醇含量(mg)','钠含量(mg)','维生素和矿物质含量','总卡路里(卡)','GI'(低,中,高)

食物分析:

血糖分析:（解释血糖为啥会这么变化）

建议:
        """,
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return agent_executor


if __name__ == "__main__":
    agent_executor = food_agent()
    result = agent_executor.invoke(
        {
            "input": "http://s7xl013pd.hn-bkt.clouddn.com/0d4c3686-282a-4829-b93f-488b57a9828e.png"
        }
    )
    print(result["output"])
