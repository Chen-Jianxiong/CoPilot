"""Validation Utilities

Utilities to validate the generated output of LLMs.
Used to verify that the tools correctly mapped questions to valid schema elements, as well as generated valid function calls.
"""

import logging
from app.log import req_id_cv
from app.tools.logwriter import LogWriter

logger = logging.getLogger(__name__)


class NoDocumentsFoundException(Exception):
    pass


class MapQuestionToSchemaException(Exception):
    pass


class InvalidFunctionCallException(Exception):
    pass


def validate_schema(conn, v_types, e_types, v_attrs, e_attrs):
    """验证LLM生成后的点边格式是否正确"""
    LogWriter.info(f"request_id={req_id_cv.get()} ENTRY validate_schema()")
    vertices = conn.getVertexTypes()
    edges = conn.getEdgeTypes()
    for v in v_types:
        logger.debug(
            f"request_id={req_id_cv.get()} validate_schema() validating vertex_type={v}"
        )
        if v in vertices:
            attrs = [x["AttributeName"] for x in conn.getVertexType(v)["Attributes"]]
            for attr in v_attrs.get(v, []):
                if attr not in attrs and attr != "":
                    raise MapQuestionToSchemaException(
                        attr
                        + " is not found for "
                        + v
                        + " in the data schema. Run MapQuestionToSchema to validate schema."
                    )
        else:
            raise MapQuestionToSchemaException(
                v
                + " is not found in the data schema. Run MapQuestionToSchema to validate schema."
            )

    for e in e_types:
        logger.debug(
            f"request_id={req_id_cv.get()} validate_schema() validating edge_type={e}"
        )
        if e in edges:
            attrs = [x["AttributeName"] for x in conn.getEdgeType(e)["Attributes"]]
            for attr in e_attrs.get(e, []):
                if attr not in attrs and attr != "":
                    raise MapQuestionToSchemaException(
                        attr
                        + " is not found for "
                        + e
                        + " in the data schema. Run MapQuestionToSchema to validate schema."
                    )
        else:
            raise MapQuestionToSchemaException(
                e
                + " is not found in the data schema. Run MapQuestionToSchema to validate schema."
            )
    LogWriter.info(f"request_id={req_id_cv.get()} EXIT validate_schema()")
    return True


def validate_function_call(conn, generated_call: str, valid_functions: list) -> str:
    """验证调用语句的正确性，返回函数名"""

    # handle installed queries
    LogWriter.info(f"request_id={req_id_cv.get()} ENTRY validate_function_call()")
    generated_call = generated_call.strip().strip("\n").strip("\t")
    # LogWriter.info(f"generated_call: {generated_call}")
    # LogWriter.info(f"valid_headers: {valid_headers}")
    # 在数据库中安装查询
    endpoints = conn.getEndpoints(dynamic=True)
    installed_queries = [q.split("/")[-1] for q in endpoints]

    # 截取函数名
    if "runInstalledQuery(" == generated_call[:18]:
        query_name = (
            generated_call.split(",")[0].split("runInstalledQuery(")[1].strip("'")
        )
        logger.debug(
            f"request_id={req_id_cv.get()} validate_function_call() validating query_name={query_name}"
        )
        logger.debug_pii(
            f"request_id={req_id_cv.get()} validate_function_call() validating query_call={generated_call}"
        )
        if query_name in valid_functions and query_name in installed_queries:
            LogWriter.info(
                f"request_id={req_id_cv.get()} EXIT validate_function_call()"
            )
            return generated_call
        elif query_name not in installed_queries:
            raise InvalidFunctionCallException(
                generated_call
                + " is not an installed function. Please select from the installed queries or install the query in the database."
            )
        else:
            raise InvalidFunctionCallException(
                generated_call
                + " is not an acceptable function. Please select from the retrieved functions."
            )
    elif "conn." == generated_call[:5]:
        return validate_function_call(
            conn, generated_call.strip("conn."), valid_functions
        )
    else:  # handle pyTG functions
        func_header = generated_call.split("(")[0]
        if (
            func_header in valid_functions
        ):  # could do more type parsing for args and things here, but will let it be for now.
            logger.debug(
                f"request_id={req_id_cv.get()} validate_function_call() validating function_header={func_header}"
            )
            logger.debug_pii(
                f"request_id={req_id_cv.get()} validate_function_call() validating function_call={generated_call}"
            )
            LogWriter.info(
                f"request_id={req_id_cv.get()} EXIT validate_function_call()"
            )
            return generated_call
        else:
            raise InvalidFunctionCallException(
                generated_call
                + " is not an acceptable function. Please select from the retrieved functions."
            )
