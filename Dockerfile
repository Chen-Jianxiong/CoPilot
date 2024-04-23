FROM python:3.11.8
WORKDIR /code

 
COPY ./requirements.txt /code/requirements.txt

 
RUN apt-get update && apt-get upgrade -y
RUN pip install -r /code/requirements.txt

 
COPY ./app /code/app

ENV LLM_CONFIG="/llm_config.json"
ENV DB_CONFIG="/db_config.json"
ENV MILVUS_CONFIG="/milvus_config.json"

# INFO, DEBUG, DEBUG_PII
ENV LOGLEVEL="INFO"


# 容器启动时执行的命令
# 启动一个在80端口上监听的uvicorn服务器，并使用app.main:app作为应用的入口点。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
