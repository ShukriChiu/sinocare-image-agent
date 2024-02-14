# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
# RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
# RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/code/app"

COPY . /code

EXPOSE 8080 

CMD exec uvicorn server:app --host 0.0.0.0 --port $PORT