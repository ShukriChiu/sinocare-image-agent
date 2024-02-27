import os
import uuid

import oss2
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from food_agent import food_agent
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_core.messages import (
    HumanMessage,
)
from PIL import Image

# from qiniu import Auth, etag, put_file

st.set_page_config(layout="wide", page_title="Sinocare Diabetes Diet Assistant")

st.title("糖尿病饮食助手")
st.subheader("上传一张图片，我们将为您分析图片中的食物并为您提供饮食建议。")

# q = Auth(
#     os.getenv("QINIU_AK"),
#     os.getenv("QINIU_SK"),
# )


_ = load_dotenv(find_dotenv(), override=True)
accessKeyId = os.getenv("OSS_ACCESS_KEY_ID")
accessKeySecret = os.getenv("OSS_ACCESS_KEY_SECRET")

# 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
auth = oss2.Auth(accessKeyId, accessKeySecret)
# yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
endpoint = "https://oss-ap-southeast-1.aliyuncs.com"

# 填写Bucket名称。
bucket = oss2.Bucket(auth, endpoint, "sinocare-food-image")
# qiniu yun
# def fix_image(upload):
#     image = Image.open(upload)
#     st.image(image)
#     image_name = str(uuid.uuid4()) + ".png"
#     image.save(image_name)
#     bucket_name = "sinocare-food-image"
#     policy = {
#         "callbackUrl": "http://food.shujian.ai/callback.php",
#         "callbackBody": "filename=$(fname)&filesize=$(fsize)",
#     }
#     token = q.upload_token(bucket_name, image_name, 360000000, policy)
#     ret, info = put_file(token, image_name, "./" + image_name, version="v2")
#     os.remove("./" + image_name)
#     response = food_agent().invoke(
#         {"input": "用户上传食物照片:http://food.shujian.ai/" + image_name}
#     )
#     st.markdown(response["output"])

# aliyun oss


def fix_image(upload):
    image = Image.open(upload)
    st.image(image)
    image_name = str(uuid.uuid4()) + ".png"
    image.save(image_name)
    bucket.put_object_from_file(image_name, "./" + image_name)
    os.remove("./" + image_name)
    response = food_agent().invoke(
        {"input": "用户上传食物照片:http://food.shujian.ai/" + image_name}
    )
    st.markdown(response["output"])


my_upload = st.file_uploader(label="上传图片", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
