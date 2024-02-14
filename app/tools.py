import os
from http import HTTPStatus
from typing import List

import dashscope
from dotenv import find_dotenv, load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
from langchain.tools import StructuredTool, tool
from langchain.vectorstores.timescalevector import TimescaleVector

# from langchain_community.embeddings import JinaEmbeddings
from langchain_openai import OpenAIEmbeddings

_ = load_dotenv(find_dotenv(), override=True)

TIMESCALE_SERVICE_URL = os.getenv("TIMESCALE_SERVICE_URL")
JINA_API_KEY = os.getenv("JINA_API_KEY")
# embeddings = JinaEmbeddings(
#     jina_api_key=JINA_API_KEY,
#     model_name="jina-embeddings-v2-base-zh",
# )

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", base_url="https://oai.hconeai.com/v1"
)


def get_device_info_tool(query: str) -> List[Document]:
    db = TimescaleVector(
        collection_name="sinocare_knowledge_base_embedding",
        service_url=TIMESCALE_SERVICE_URL,
        embedding=embeddings,
    )
    compressor = CohereRerank(model="rerank-multilingual-v2.0", top_n=1)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.3},
        ),
    )

    return compression_retriever.get_relevant_documents(query)


get_device_info = StructuredTool.from_function(
    func=get_device_info_tool,
    name="device_info",
    description="Useful for retrieving device information, contains common FAQ around deiivces, such as working temerature, battery life, accuracy, etc.",
    handle_tool_error=True,
)


def get_user_profile_info_tool(user_id: str) -> str:
    return """
{
            "用户名": "罗和平",
            "年龄": "70",
            "性别": "男",
            "身高": "163.5cm",
            "体重": "67.15kg",
            "腰围": "94cm",
            "糖尿病类型": "2型",
            "确诊时间": "2003-10-29",
            "最近HbA1c": "11.8 (2023-10-29)",
            "主要症状": "患者10年前发现血糖高，诊断为2型糖尿病，患者现予以二甲双胍 1片 每天一次 及达美康 2片 每天一次治疗，有口干、尿多，有视物模糊、手脚麻木，夜间睡眠可，皮肤瘙痒感，双下肢怕冷，头晕、头痛。",
            "既往病史": "过敏性鼻炎、肾结石",
            "药物过敏史": "无",
            "流行病史": "无",
            "个人史": "抽烟，一天10根",
            "家族史": "无",
}
"""


get_user_profile_info = StructuredTool.from_function(
    func=get_user_profile_info_tool,
    name="user_profile_info",
    description="Useful for retrieving user profile information, contains healthcare profile infomation for users",
    handle_tool_error=True,
)


def get_user_status_info_tool(user_id: str) -> str:
    return """
{
    "是否正在佩戴CGM": "是",
    "佩戴的第几天": "第3天",
}
"""


get_user_status_info = StructuredTool.from_function(
    func=get_user_status_info_tool,
    name="user_status_info",
    description="Useful for retrieving user status information, contains device usage information or other user status information",
    handle_tool_error=True,
)


def get_user_glucose_info_tool(user_id: str) -> str:
    return ""


get_user_glucose_info = StructuredTool.from_function(
    func=get_user_glucose_info_tool,
    name="user_glucose_info",
    description="Useful for retrieving user glucose information",
    handle_tool_error=True,
)


def get_user_preference_info_tool(user_id: str) -> str:
    return "{'like':'足球，游泳，篮球，坚果','dislike':'奶制品'}"


get_user_preference_info = StructuredTool.from_function(
    func=get_user_preference_info_tool,
    name="user_preference",
    description="Useful for retrieving user preference information, related to food choice, exercise choice, etc. mainly in lifestyle perspective",
    handle_tool_error=True,
)


def get_drug_info_tool(query: str) -> List[Document]:
    db = TimescaleVector(
        collection_name="sinocare_drug_info_embedding",
        service_url=TIMESCALE_SERVICE_URL,
        embedding=embeddings,
    )
    compressor = CohereRerank()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.5},
        ),
    )

    return compression_retriever.get_relevant_documents(query)


get_drug_info = StructuredTool.from_function(
    func=get_drug_info_tool,
    name="drug_info",
    description="Useful for retrieving drug information",
    handle_tool_error=True,
)


def get_food_info_tool(query: str) -> List[Document]:
    # Your code here
    return []


get_food_info = StructuredTool.from_function(
    func=get_food_info_tool,
    name="food_info",
    description="Useful for retrieving food information",
    handle_tool_error=True,
)


def get_exercise_info_tool(query: str) -> List[Document]:
    # Your code here
    return []


get_exercise_info = StructuredTool.from_function(
    func=get_exercise_info_tool,
    name="exercise_info",
    description="Useful for retrieving exercise information",
    handle_tool_error=True,
)


@tool
def image_description(image_url: str, query: str):
    """userful when you needs to deal with a image url, you need to construct query to describe image or extract infomation from image"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": query},
            ],
        }
    ]
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    response = dashscope.MultiModalConversation.call(
        model="qwen-vl-max", messages=messages
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message["content"][0]["text"]
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.


# print(
#     image_description.run(
#         {
#             "image_url": "http://s7xl013pd.hn-bkt.clouddn.com/2568afe6-c8eb-40fc-87f8-cbbf9669c10d.png",
#             "query": "Describe the main food items and their sizes in the image.",
#         }
#     )
# )
# print(get_drug_info.run("metformin"))

# print(get_device_info.run("存储温度"))
