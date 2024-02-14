# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
# RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
# RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# 定义构建时变量
ARG LANGCHAIN_TRACING_V2
ARG LANGCHAIN_ENDPOINT
ARG LANGCHAIN_API_KEY
ARG LANGCHAIN_PROJECT
ARG OPENAI_API_KEY
ARG TIMESCALE_SERVICE_URL
ARG HELICONE_API
ARG REDIS_URL
ARG AZURE_OPENAI_API_KEY_INDIA
ARG AZURE_OPENAI_ENDPOINT_INDIA
ARG AZURE_OPENAI_API_KEY_AUS
ARG AZURE_OPENAI_ENDPOINT_AUS
ARG AZURE_OPENAI_API_KEY_EASTUS1
ARG AZURE_OPENAI_ENDPOINT_EASTUS1
ARG AZURE_OPENAI_API_KEY_WESTUS
ARG AZURE_OPENAI_ENDPOINT_WESTUS
ARG AZURE_OPENAI_API_VERSION
ARG SERPER_API_KEY
ARG TAVILY_API_KEY
ARG COHERE_API_KEY
ARG HUGGINGFACEHUB_API_TOKEN
ARG MINIMAX_GROUP_ID
ARG MINIMAX_API_KEY
ARG JINA_API_KEY
ARG MINIMAX_API_HOST
ARG SCENEX_API_KEY
ARG DASHSCOPE_API_KEY
ARG MOONSHOT_API_KEY
ARG QINIU_AK
ARG QINIU_SK


COPY . /code

EXPOSE 8080 

CMD exec uvicorn app.server:app --host 0.0.0.0 --port $PORT