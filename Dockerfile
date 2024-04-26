# 使用 Python 3.11.8 作为基础镜像
FROM python:3.11.8

# 将容器的工作目录设置为/code
# （通过docker exec -it your_docker_container_id bash 进入容器后的默认目录）。
WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

 
RUN apt-get update && apt-get upgrade -y
RUN pip install -r /code/requirements.txt

 
COPY ./app /code/app

ENV LLM_CONFIG="/llm_config.json"
ENV DB_CONFIG="/db_config.json"
ENV MILVUS_CONFIG="/milvus_config.json"

# 日志级别：INFO, DEBUG, DEBUG_PII
ENV LOGLEVEL="INFO"

# 容器启动时执行的命令
# 启动一个在8000端口上监听的uvicorn服务器，并使用app.main:app作为应用的入口点。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
