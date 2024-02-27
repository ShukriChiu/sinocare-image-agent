# -*- coding: utf-8 -*-
import os

import oss2
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv(), override=True)
accessKeyId = os.getenv("OSS_ACCESS_KEY_ID")
accessKeySecret = os.getenv("OSS_ACCESS_KEY_SECRET")

# 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
auth = oss2.Auth(accessKeyId, accessKeySecret)
# yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
endpoint = "https://oss-ap-southeast-1.aliyuncs.com"

# 填写Bucket名称。
bucket = oss2.Bucket(auth, endpoint, "sinocare-food-image")
# oss2.ObjectIterator用于遍历文件。
# 上传Object。
result = bucket.put_object_from_file(
    "test.png", "/root/shujiancoding/sinocare-image-agent/test.jpeg"
)
print(result)
