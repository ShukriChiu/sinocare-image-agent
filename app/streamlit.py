import os
import uuid

import streamlit as st
from food_agent import food_agent
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_core.messages import (
    HumanMessage,
)
from PIL import Image
from qiniu import Auth, etag, put_file

st.set_page_config(layout="wide", page_title="Sinocare Diabetes Diet Assistant")

st.title("糖尿病饮食助手")
st.subheader("上传一张图片，我们将为您分析图片中的食物并为您提供饮食建议。")

q = Auth(
    os.getenv("QINIU_AK"),
    os.getenv("QINIU_SK"),
)


def fix_image(upload):
    image = Image.open(upload)
    st.image(image)
    image_name = str(uuid.uuid4()) + ".png"
    image.save(image_name)
    bucket_name = "sinocare-image"
    policy = {
        "callbackUrl": "http://s7xl013pd.hn-bkt.clouddn.com/callback.php",
        "callbackBody": "filename=$(fname)&filesize=$(fsize)",
    }
    token = q.upload_token(bucket_name, image_name, 360000000, policy)
    ret, info = put_file(token, image_name, "./" + image_name, version="v2")
    os.remove("./" + image_name)
    response = food_agent().invoke(
        {
            "messages": [
                HumanMessage(
                    content="http://s7xl013pd.hn-bkt.clouddn.com/" + image_name
                )
            ]
        }
    )
    st.markdown(response["output"])


my_upload = st.file_uploader(label="上传图片", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
