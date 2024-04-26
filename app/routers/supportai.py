import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.config import (embedding_service, embedding_store, get_llm_service,
                        llm_config)

from app.py_schemas.schemas import (CoPilotResponse, CreateIngestConfig,
                                    LoadingInfo, SupportAIQuestion)
from app.supportai.concept_management.create_concepts import (
    CommunityConceptCreator, EntityConceptCreator, HigherLevelConceptCreator,
    RelationshipConceptCreator)
from app.supportai.retrievers import (EntityRelationshipRetriever,
                                      HNSWOverlapRetriever, HNSWRetriever,
                                      HNSWSiblingRetriever)

from app.util import get_eventual_consistency_checker

"""
开发中，即将发布 alpha 版本
摄取一组文档，从信息中提取知识图，并通过自然语言查询实现文档和图数据的混合搜索。
此功能将利用图形数据丰富 RAG（检索增强生成）管道，从而能够对用户查询做出更准确、信息更丰富的响应。 
"""
logger = logging.getLogger(__name__)
router = APIRouter(tags=["SupportAI"])


@router.post("/{graphname}/supportai/initialize")
def initialize(graphname, conn: Request):
    """ 读取gsql文件，并使用conn.gsql()执行gsql文件中的语句。以初始化supportai """
    # 需要使用绝对路径打开文件
    # 定义了一个schema change job，其主要目的是在图数据库中添加一系列新的顶点类型和边类型。

    conn = conn.state.conn
    # need to open the file using the absolute path
    file_path = "app/gsql/supportai/SupportAI_Schema.gsql"
    with open(file_path, "r") as f:
        schema = f.read()
    # 运行一个名为add_supportai_schema的schema变更job。
    schema_res = conn.gsql(
        """USE GRAPH {}\n{}\nRUN SCHEMA_CHANGE JOB add_supportai_schema""".format(
            graphname, schema
        )
    )

    # 在多个顶点类型上添加索引，以优化基于时间戳字段的查询性能。
    file_path = "app/gsql/supportai/SupportAI_IndexCreation.gsql"
    with open(file_path) as f:
        index = f.read()
    index_res = conn.gsql(
        """USE GRAPH {}\n{}\nRUN SCHEMA_CHANGE JOB add_supportai_indexes""".format(
            graphname, index
        )
    )

    # 扫描指定类型的顶点，并根据它们的时间戳属性确定哪些顶点需要处理或更新；同时提取这些顶点的相关文本或定义
    file_path = "app/gsql/supportai/Scan_For_Updates.gsql"
    with open(file_path) as f:
        scan_for_updates = f.read()
    res = conn.gsql(
        "USE GRAPH "
        + conn.graphname
        + "\n"
        + scan_for_updates
        + "\n INSTALL QUERY Scan_For_Updates"
    )

    # 分布式查询，更新一组特定顶点的处理状态。修改这些顶点的时间戳属性，标记它们为已处理状态。
    file_path = "app/gsql/supportai/Update_Vertices_Processing_Status.gsql"
    with open(file_path) as f:
        update_vertices = f.read()
    res = conn.gsql(
        "USE GRAPH "
        + conn.graphname
        + "\n"
        + update_vertices
        + "\n INSTALL QUERY Update_Vertices_Processing_Status"
    )

    return {
        # 包括host_name以便从客户端进行调试。它们的pyTG conn可能与在copilot中配置的主机不同
        "host_name": conn._tg_connection.host,
        "schema_creation_status": json.dumps(schema_res),
        "index_creation_status": json.dumps(index_res)
    }


@router.post("/{graphname}/supportai/create_ingest")
def create_ingest(
    graphname,
    ingest_config: CreateIngestConfig,
    conn: Request
):
    """ 摄取文档前的准备工作 """
    conn = conn.state.conn

    if ingest_config.file_format.lower() == "json":
        # 定义了一个数据加载作业，用于从 JSON 格式的文件中导入数据到图数据库中的顶点和边。
        file_path = "app/gsql/supportai/SupportAI_InitialLoadJSON.gsql"

        with open(file_path) as f:
            ingest_template = f.read()
        ingest_template = ingest_template.replace("@uuid@", str(uuid.uuid4().hex))
        doc_id = ingest_config.loader_config.get("doc_id_field", "doc_id")
        doc_text = ingest_config.loader_config.get("content_field", "content")
        ingest_template = ingest_template.replace('"doc_id"', '"{}"'.format(doc_id))
        ingest_template = ingest_template.replace('"content"', '"{}"'.format(doc_text))

    if ingest_config.file_format.lower() == "csv":
        # 定义了一个数据加载作业，旨在从一个 CSV 文件中导入数据到图数据库的顶点和边中。
        file_path = "app/gsql/supportai/SupportAI_InitialLoadCSV.gsql"

        with open(file_path) as f:
            ingest_template = f.read()
        ingest_template = ingest_template.replace("@uuid@", str(uuid.uuid4().hex))
        separator = ingest_config.get("separator", "|")
        header = ingest_config.get("header", "true")
        eol = ingest_config.get("eol", "\n")
        quote = ingest_config.get("quote", "double")
        ingest_template = ingest_template.replace('"|"', '"{}"'.format(separator))
        ingest_template = ingest_template.replace('"true"', '"{}"'.format(header))
        ingest_template = ingest_template.replace('"\\n"', '"{}"'.format(eol))
        ingest_template = ingest_template.replace('"double"', '"{}"'.format(quote))

    # 在图数据库环境中注册一个外部数据源，使其可用于之后的数据加载或集成任务。
    file_path = "app/gsql/supportai/SupportAI_DataSourceCreation.gsql"

    with open(file_path) as f:
        data_stream_conn = f.read()

    # 为数据流连接分配唯一标识符
    data_stream_conn = data_stream_conn.replace(
        "@source_name@", "SupportAI_" + graphname + "_" + str(uuid.uuid4().hex)
    )

    # 检查数据源并创建适当的连接
    if ingest_config.data_source.lower() == "s3":
        data_conn = ingest_config.data_source_config
        if (
            data_conn.get("aws_access_key") is None
            or data_conn.get("aws_secret_key") is None
        ):
            raise Exception("AWS credentials not provided")
        connector = {
            "type": "s3",
            "access.key": data_conn["aws_access_key"],
            "secret.key": data_conn["aws_secret_key"],
        }

        data_stream_conn = data_stream_conn.replace(
            "@source_config@", json.dumps(connector)
        )

    elif ingest_config.data_source.lower() == "azure":
        if ingest_config.data_source_config.get("account_key") is not None:
            connector = {
                "type": "abs",
                "account.key": ingest_config.data_source_config["account_key"],
            }
        elif ingest_config.data_source_config.get("client_id") is not None:
            # 验证是否还提供了客户端密钥
            if ingest_config.data_source_config.get("client_secret") is None:
                raise Exception("Client secret not provided")
            # 验证是否还提供了租户id
            if ingest_config.data_source_config.get("tenant_id") is None:
                raise Exception("Tenant id not provided")
            connector = {
                "type": "abs",
                "client.id": ingest_config.data_source_config["client_id"],
                "client.secret": ingest_config.data_source_config["client_secret"],
                "tenant.id": ingest_config.data_source_config["tenant_id"],
            }
        else:
            raise Exception("Azure credentials not provided")
        data_stream_conn = data_stream_conn.replace(
            "@source_config@", json.dumps(connector)
        )
    elif ingest_config.data_source.lower() == "gcs":
        # 验证是否提供了正确的字段
        if ingest_config.data_source_config.get("project_id") is None:
            raise Exception("Project id not provided")
        if ingest_config.data_source_config.get("private_key_id") is None:
            raise Exception("Private key id not provided")
        if ingest_config.data_source_config.get("private_key") is None:
            raise Exception("Private key not provided")
        if ingest_config.data_source_config.get("client_email") is None:
            raise Exception("Client email not provided")
        connector = {
            "type": "gcs",
            "project_id": ingest_config.data_source_config["project_id"],
            "private_key_id": ingest_config.data_source_config["private_key_id"],
            "private_key": ingest_config.data_source_config["private_key"],
            "client_email": ingest_config.data_source_config["client_email"],
        }
        data_stream_conn = data_stream_conn.replace(
            "@source_config@", json.dumps(connector)
        )
    else:
        raise Exception("Data source not implemented")

    load_job_created = conn.gsql("USE GRAPH {}\n".format(graphname) + ingest_template)

    data_source_created = conn.gsql(
        "USE GRAPH {}\n".format(graphname) + data_stream_conn
    )

    return {
        "load_job_id": load_job_created.split(":")[1]
        .strip(" [")
        .strip(" ")
        .strip(".")
        .strip("]"),
        "data_source_id": data_source_created.split(":")[1]
        .strip(" [")
        .strip(" ")
        .strip(".")
        .strip("]"),
    }


@router.post("/{graphname}/supportai/ingest")
def ingest(
    graphname,
    loader_info: LoadingInfo,
    background_tasks: BackgroundTasks,
    conn: Request
):
    conn = conn.state.conn
    # 添加一个在发送响应后在后台调用的函数（最终一致性检查器）。
    background_tasks.add_task(get_eventual_consistency_checker, graphname, conn)
    if loader_info.file_path is None:
        raise Exception("File path not provided")
    if loader_info.load_job_id is None:
        raise Exception("Load job id not provided")
    if loader_info.data_source_id is None:
        raise Exception("Data source id not provided")

    try:
        # 加载作业
        res = conn.gsql(
            'USE GRAPH {}\nRUN LOADING JOB -noprint {} USING {}="{}"'.format(
                graphname,
                loader_info.load_job_id,
                "DocumentContent",
                "$" + loader_info.data_source_id + ":" + loader_info.file_path,
            )
        )
    except Exception as e:
        if (
            "Running the following loading job in background with '-noprint' option:"
            in str(e)
        ):
            res = str(e)
        else:
            raise e
    return {
        "job_name": loader_info.load_job_id,
        "job_id": res.split(
            "Running the following loading job in background with '-noprint' option:"
        )[1]
        .split("Jobid: ")[1]
        .split("\n")[0],
        "log_location": res.split(
            "Running the following loading job in background with '-noprint' option:"
        )[1]
        .split("Log directory: ")[1]
        .split("\n")[0],
    }


@router.post("/{graphname}/supportai/search")
def search(
    graphname,
    query: SupportAIQuestion,
    conn: Request
):
    """
    搜索，根据提问生成搜索方法，并执行搜索与之相关的tok_k信息
    """
    conn = conn.state.conn
    if query.method.lower() == "hnswoverlap":
        # 创建基于 HNSW 索引的 overlap 搜索对象
        retriever = HNSWOverlapRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        res = retriever.search(
            query.question,
            query.method_params["indices"],
            query.method_params["top_k"],
            query.method_params["num_hops"],
            query.method_params["num_seen_min"],
        )
    elif query.method.lower() == "vdb":
        if "index" not in query.method_params:
            raise Exception("Index name not provided")
        # 创建基于 HNSW 索引的搜索对象
        retriever = HNSWRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        res = retriever.search(
            query.question,
            query.method_params["index"],
            query.method_params["top_k"],
            # 是否使用 HyDE 方法
            query.method_params["withHyDE"],
        )
    elif query.method.lower() == "sibling":
        if "index" not in query.method_params:
            raise Exception("Index name not provided")
        # 创建基于 HNSW 索引的 Sibling 搜索对象
        retriever = HNSWSiblingRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        res = retriever.search(
            query.question,
            query.method_params["index"],
            query.method_params["top_k"],
            query.method_params["lookback"],
            query.method_params["lookahead"],
            query.method_params["withHyDE"],
        )
    elif query.method.lower() == "entityrelationship":
        # 实体关系检索器
        retriever = EntityRelationshipRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        # 执行检索，使用llm从提问中提取实体关系，再由tg执行查询语句检索与之相关的top_K片段
        res = retriever.search(query.question, query.method_params["top_k"])

    return res


@router.post("/{graphname}/supportai/answerquestion")
def answer_question(
    graphname,
    query: SupportAIQuestion,
    conn: Request
):
    """
    与search相同的执行，在其结果上由llm生成回答
    """
    conn = conn.state.conn
    resp = CoPilotResponse
    resp.response_type = "supportai"
    if query.method.lower() == "hnswoverlap":
        # 基于 HNSW 索引的 overlap 搜索。
        retriever = HNSWOverlapRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        # 先在图中搜索，再由llm生成回答
        res = retriever.retrieve_answer(
            query.question,
            query.method_params["indices"],
            query.method_params["top_k"],
            query.method_params["num_hops"],
            query.method_params["num_seen_min"],
        )
    elif query.method.lower() == "vdb":
        if "index" not in query.method_params:
            raise Exception("Index name not provided")
        # 基于 HNSW 索引的搜索。
        retriever = HNSWRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        res = retriever.retrieve_answer(
            query.question,
            query.method_params["index"],
            query.method_params["top_k"],
            query.method_params["withHyDE"],
        )
    elif query.method.lower() == "sibling":
        if "index" not in query.method_params:
            raise Exception("Index name not provided")
        retriever = HNSWSiblingRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        # 先在图中搜索，再由llm生成回答
        res = retriever.retrieve_answer(
            query.question,
            query.method_params["index"],
            query.method_params["top_k"],
            query.method_params["lookback"],
            query.method_params["lookahead"],
            query.method_params["withHyDE"],
        )
    elif query.method.lower() == "entityrelationship":
        # 实体关系检索器
        retriever = EntityRelationshipRetriever(
            embedding_service, embedding_store, get_llm_service(llm_config), conn
        )
        # 先使用llm提取关系实体，并过滤，再由llm生成回答
        res = retriever.retrieve_answer(query.question, query.method_params["top_k"])
    else:
        raise Exception("Method not implemented")

    resp.natural_language_response = res["response"]
    resp.query_sources = res["retrieved"]

    return res


@router.get("/{graphname}/supportai/buildconcepts")
def build_concepts(
    graphname,
    background_tasks: BackgroundTasks,
    conn: Request,
):
    """
    创建相关概念
    """
    conn = conn.state.conn
    background_tasks.add_task(get_eventual_consistency_checker, graphname)
    # 边
    rels_concepts = RelationshipConceptCreator(conn, llm_config, embedding_service)
    rels_concepts.create_concepts()
    # 点
    ents_concepts = EntityConceptCreator(conn, llm_config, embedding_service)
    ents_concepts.create_concepts()
    # 社区
    comm_concepts = CommunityConceptCreator(conn, llm_config, embedding_service)
    comm_concepts.create_concepts()
    # 共现关系、概念树
    high_level_concepts = HigherLevelConceptCreator(conn, llm_config, embedding_service)
    high_level_concepts.create_concepts()

    return {"status": "success"}


@router.get("/{graphname}/supportai/forceupdate")
async def force_update(
    graphname: str, conn: Request
):
    """
    强制更新，执行最终一次性检查器，协调 Milvus 和 TigerGraph 数据
    """
    conn = conn.state.conn
    get_eventual_consistency_checker(graphname, conn)
    return {"status": "success"}
