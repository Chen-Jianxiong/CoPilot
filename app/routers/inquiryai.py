import json
import logging
import traceback
from typing import Annotated, List, Union, Optional

from fastapi import (APIRouter, Depends, HTTPException, Request, WebSocket,
                     status)
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasicCredentials

from app.agent import TigerGraphAgent
from app.config import (embedding_service, embedding_store, llm_config,
                        security, session_handler)
from app.llm_services import (AWS_SageMaker_Endpoint, AWSBedrock, AzureOpenAI,
                              GoogleVertexAI, OpenAI)
from app.log import req_id_cv
from app.metrics.prometheus_metrics import metrics as pmetrics
from app.metrics.tg_proxy import TigerGraphConnectionProxy
from app.py_schemas.schemas import (CoPilotResponse, GSQLQueryInfo, GSQLQueryList,
                                    NaturalLanguageQuery, QueryDeleteRequest,
                                    QueryUpsertRequest)
from app.tools.logwriter import LogWriter
from app.tools.validation_utils import MapQuestionToSchemaException

"""
自然语言查询服务，允许用户用简单的英语询问有关其图形数据的问题。
该服务使用大型语言模型（LLM）将用户的问题转换为函数调用，然后在图形数据库上执行。
"""

logger = logging.getLogger(__name__)
router = APIRouter(tags=["InquiryAI"])


@router.post("/{graphname}/query")
def retrieve_answer(
    graphname,
    query: NaturalLanguageQuery,
    conn: Request,
) -> CoPilotResponse:
    """
    检索给定问题的答案。
    conn: 一个PyTigerGraph TigerGraphConnection对象实例化，以与所需的数据库/图交互，并使用正确的角色进行身份验证。
    """
    conn = conn.state.conn
    logger.debug_pii(
        f"/{graphname}/query request_id={req_id_cv.get()} question={query.query}"
    )
    logger.debug(
        f"/{graphname}/query request_id={req_id_cv.get()} database connection created"
    )

    # 创建llm相应的 代理执行器，其中包含了mq2s和gen_func
    if llm_config["completion_service"]["llm_service"].lower() == "openai":
        logger.debug(
            f"/{graphname}/query request_id={req_id_cv.get()} llm_service=openai agent created"
        )
        agent = TigerGraphAgent(
            OpenAI(llm_config["completion_service"]),
            conn,
            embedding_service,
            embedding_store,
        )
    elif llm_config["completion_service"]["llm_service"].lower() == "azure":
        logger.debug(
            f"/{graphname}/query request_id={req_id_cv.get()} llm_service=azure agent created"
        )
        agent = TigerGraphAgent(
            AzureOpenAI(llm_config["completion_service"]),
            conn,
            embedding_service,
            embedding_store,
        )
    elif llm_config["completion_service"]["llm_service"].lower() == "sagemaker":
        logger.debug(
            f"/{graphname}/query request_id={req_id_cv.get()} llm_service=sagemaker agent created"
        )
        agent = TigerGraphAgent(
            AWS_SageMaker_Endpoint(llm_config["completion_service"]),
            conn,
            embedding_service,
            embedding_store,
        )
    elif llm_config["completion_service"]["llm_service"].lower() == "vertexai":
        logger.debug(
            f"/{graphname}/query request_id={req_id_cv.get()} llm_service=vertexai agent created"
        )
        agent = TigerGraphAgent(
            GoogleVertexAI(llm_config["completion_service"]),
            conn,
            embedding_service,
            embedding_store,
        )
    elif llm_config["completion_service"]["llm_service"].lower() == "bedrock":
        logger.debug(
            f"/{graphname}/query request_id={req_id_cv.get()} llm_service=bedrock agent created"
        )
        agent = TigerGraphAgent(
            AWSBedrock(llm_config["completion_service"]),
            conn,
            embedding_service,
            embedding_store,
        )
    else:
        LogWriter.error(
            f"/{graphname}/query request_id={req_id_cv.get()} agent creation failed due to invalid llm_service"
        )
        raise Exception("LLM Completion Service Not Supported")

    resp = CoPilotResponse(
        natural_language_response="", answered_question=False, response_type="inquiryai"
    )
    steps = ""
    try:
        # 调用代理处理问题，返回结果
        steps = agent.question_for_agent(query.query)
        # 如果没有采取任何步骤，请再试一次
        if len(steps["intermediate_steps"]) == 0:
            steps = agent.question_for_agent(query.query)

        logger.debug(f"/{graphname}/query request_id={req_id_cv.get()} agent executed")
        try:
            # 修饰响应结果
            generate_func_output = steps["intermediate_steps"][-1][-1]
            resp.natural_language_response = steps["output"]
            resp.query_sources = {
                "function_call": generate_func_output["function_call"],
                "result": json.loads(generate_func_output["result"]),
                "reasoning": generate_func_output["reasoning"],
            }
            resp.answered_question = True
            pmetrics.llm_success_response_total.labels(
                embedding_service.model_name
            ).inc()
        except Exception:
            resp.natural_language_response = (
                # "An error occurred while processing the response. Please try again."
                str(steps["output"])
            )
            resp.query_sources = {"agent_history": str(steps)}
            resp.answered_question = False
            LogWriter.warning(
                f"/{graphname}/query request_id={req_id_cv.get()} agent execution failed due to unknown exception"
            )
            pmetrics.llm_query_error_total.labels(embedding_service.model_name).inc()
            # 打印错误日志
            exc = traceback.format_exc()
            logger.debug_pii(
                f"/{graphname}/query request_id={req_id_cv.get()} Exception Trace:\n{exc}"
            )
    except MapQuestionToSchemaException:
        resp.natural_language_response = (
            "A schema mapping error occurred. Please try rephrasing your question."
        )
        resp.query_sources = {}
        resp.answered_question = False
        LogWriter.warning(
            f"/{graphname}/query request_id={req_id_cv.get()} agent execution failed due to MapQuestionToSchemaException"
        )
        pmetrics.llm_query_error_total.labels(embedding_service.model_name).inc()
        exc = traceback.format_exc()
        logger.debug_pii(
            f"/{graphname}/query request_id={req_id_cv.get()} Exception Trace:\n{exc}"
        )
    except Exception as e:
        resp.natural_language_response = (
            # "An error occurred while processing the response. Please try again."
            str(steps["output"])
        )
        resp.query_sources = {} if len(steps) == 0 else {"agent_history": str(steps)}
        resp.answered_question = False
        LogWriter.warning(
            f"/{graphname}/query request_id={req_id_cv.get()} agent execution failed due to unknown exception"
        )
        exc = traceback.format_exc()
        logger.debug_pii(
            f"/{graphname}/query request_id={req_id_cv.get()} Exception Trace:\n{exc}"
        )
        pmetrics.llm_query_error_total.labels(embedding_service.model_name).inc()

    return resp

@router.get("/{graphname}/list_registered_queries")
def list_registered_queries(graphname, conn: Request):
    conn = conn.state.conn
    if conn.getVer().split(".")[0] <= "3":
        query_descs = embedding_store.list_registered_documents(graphname=graphname, only_custom=True, output_fields=["function_header", "text"])
    else:
        queries = embedding_store.list_registered_documents(graphname=graphname, only_custom=True, output_fields=["function_header"])
        if not queries:
            return {"queries": {}}
        query_descs = conn.getQueryDescription([x["function_header"] for x in queries])

    return query_descs


@router.post("/{graphname}/getqueryembedding")
def get_query_embedding(
    graphname,
    query: NaturalLanguageQuery
):
    logger.debug(
        f"/{graphname}/getqueryembedding request_id={req_id_cv.get()} question={query.query}"
    )

    # 为查询文本生成长度安全的嵌入向量
    return embedding_service.embed_query(query.query)


@router.post("/{graphname}/register_docs")
def register_docs(
    graphname,
    query_list: Union[GSQLQueryInfo, List[GSQLQueryInfo]],
    conn: Request
):
    """注册定制查询方法"""
    conn = conn.state.conn
    # auth check
    try:
        conn.echo()
    except Exception as e:
        raise HTTPException(
            status_code=401, detail="Invalid credentials"
        )
    logger.debug(f"Using embedding store: {embedding_store}")
    results = []

    if not isinstance(query_list, list):
        query_list = [query_list]

    for query_info in query_list:
        logger.debug(
            f"/{graphname}/register_docs request_id={req_id_cv.get()} registering {query_info.function_header}"
        )

        # 为查询文本生成长度安全的嵌入向量
        vec = embedding_service.embed_query(query_info.docstring)
        # 添加到向量存储中
        res = embedding_store.add_embeddings(
            [(query_info.docstring, vec)],
            [
                {
                    "function_header": query_info.function_header,
                    "description": query_info.description,
                    "param_types": query_info.param_types,
                    "custom_query": True,
                    "graphname": query_info.graphname
                }
            ],
        )
        if res:
            results.append(res)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register document(s)",
            )

    return results

@router.post("/{graphname}/upsert_from_gsql")
def upsert_from_gsql(
    graphname,
    query_list: GSQLQueryList,
    conn: Request
):
    """ 根据传入的gsql查询列表，更新定制查询方法 """
    conn = conn.state.conn

    query_names = query_list.queries
    # 获取查询的描述信息
    query_descs = conn.getQueryDescription(query_names)

    query_info_list = []
    for query_desc in query_descs:
        print(query_desc)
        params = query_desc["parameters"]
        if params == []:
            params = {}
        else:
            tmp_params = {}
            for param in params:
                tmp_params[param["paramName"]] = "INSERT " + param.get("description", "VALUE") + " HERE"
            params = tmp_params
        param_types = conn.getQueryMetadata(query_desc["queryName"])["input"]
        q_info = GSQLQueryInfo(
                function_header=query_desc["queryName"],
                description=query_desc["description"],
                docstring=query_desc["description"]+ ".\nRun with runInstalledQuery('"+query_desc["queryName"]+"', params={})".format(json.dumps(params)),
                param_types={list(x.keys())[0]: x[list(x.keys())[0]] for x in param_types},
                graphname=graphname
            )

        query_info_list.append(QueryUpsertRequest(id=None, query_info=q_info))
    return upsert_docs(graphname, query_info_list)

@router.post("/{graphname}/delete_from_gsql")
def delete_from_gsql(
    graphname,
    query_list: GSQLQueryList,
    conn: Request
):
    """ 根据传入的gsql查询列表，删除定制查询方法 """
    conn = conn.state.conn

    query_names = query_list.queries
    query_descs = conn.getQueryDescription(query_names)

    func_counter = 0

    for query_desc in query_descs:
        delete_docs(graphname, QueryDeleteRequest(ids=None, expr=f"function_header=='{query_desc['queryName']}' and graphname=='{graphname}'"))
        func_counter += 1

    return {"deleted_functions": query_descs, "deleted_count": func_counter}


@router.post("/{graphname}/upsert_docs")
def upsert_docs(
    graphname,
    request_data: Union[QueryUpsertRequest, List[QueryUpsertRequest]],
    conn: Request
):
    """ 更新定制查询方法 """
    conn = conn.state.conn
    # auth check
    try:
        conn.echo()
    except Exception as e:
        raise HTTPException(
            status_code=401, detail="Invalid credentials"
        )
    try:
        results = []

        if not isinstance(request_data, list):
            request_data = [request_data]

        for request_info in request_data:
            id = request_info.id
            query_info = request_info.query_info

            if not id and not query_info:
                raise HTTPException(
                    status_code=400,
                    detail="At least one of 'id' or 'query_info' is required",
                )

            logger.debug(
                f"/{graphname}/upsert_docs request_id={req_id_cv.get()} upserting document(s)"
            )

            # 为查询文本生成长度安全的嵌入向量
            vec = embedding_service.embed_query(query_info.docstring)
            # 更新到向量存储中
            res = embedding_store.upsert_embeddings(
                id,
                [(query_info.docstring, vec)],
                [
                    {
                        "function_header": query_info.function_header,
                        "description": query_info.description,
                        "param_types": query_info.param_types,
                        "custom_query": True,
                        "graphname": query_info.graphname
                    }
                ],
            )
            if res:
                results.append(res)
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to upsert document(s)",
                )
        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred while upserting query {str(e)}"
        )


@router.post("/{graphname}/delete_docs")
def delete_docs(graphname, request_data: QueryDeleteRequest, conn: Request):
    """ 删除定制查询方法 """

    conn = conn.state.conn
    # auth check
    try:
        conn.echo()
    except Exception as e:
        raise HTTPException(
            status_code=401, detail="Invalid credentials"
        )
    ids = request_data.ids
    expr = request_data.expr

    if ids and not isinstance(ids, list):
        try:
            ids = [ids]
        except ValueError:
            raise ValueError(
                "Invalid ID format. IDs must be string or lists of strings."
            )

    logger.debug(
        f"/{graphname}/delete_docs request_id={req_id_cv.get()} deleting document(s)"
    )

    # 根据提供的id或表达式调用remove_embeddings方法
    try:
        if expr:
            res = embedding_store.remove_embeddings(expr=expr)
            return res
        elif ids:
            res = embedding_store.remove_embeddings(ids=ids)
            return res
        else:
            raise HTTPException(
                status_code=400, detail="Either IDs or an expression must be provided."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{graphname}/retrieve_docs")
def retrieve_docs(
    graphname,
    query: NaturalLanguageQuery,
    top_k: int = 3,
):
    """ 从给定查询嵌入的向量存储中检索top_k相似的嵌入。"""
    logger.debug_pii(
        f"/{graphname}/retrieve_docs request_id={req_id_cv.get()} top_k={top_k} question={query.query}"
    )
    return embedding_store.retrieve_similar(
        embedding_service.embed_query(query.query), top_k=top_k
    )


@router.post("/{graphname}/login")
def login(graphname, conn: Request):
    session_id = session_handler.create_session(conn.state.conn.username, conn)
    return {"session_id": session_id}


@router.post("/{graphname}/logout")
def logout(graphname, session_id: str):
    session_handler.delete_session(session_id)
    return {"status": "success"}


@router.get("/{graphname}/chat")
def chat(request: Request):
    """读取chat.html文件的内容，并将其作为HTML响应返回。"""
    return HTMLResponse(open("app/static/chat.html").read())



@router.websocket("/{graphname}/ws")
async def websocket_endpoint(websocket: WebSocket, graphname: str, session_id: str):
    """实现一个基于WebSocket的API，WebSocket 协议使得数据可以在用户和服务器之间双向传输"""
    session = session_handler.get_session(session_id)
    # 获取与客户端连接的WebSocket对象，并调用accept()方法接受连接。
    await websocket.accept()
    while True:
        # 持续监听客户端发送的消息
        data = await websocket.receive_text()
        # 回答
        res = retrieve_answer(
            graphname, NaturalLanguageQuery(query=data), session.db_conn
        )
        # 将响应发送回客户端
        await websocket.send_text(f"{res.natural_language_response}")
