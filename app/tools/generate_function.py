import json
import logging
from typing import Dict, List, Optional, Type, Union

from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import BaseTool
from langchain.tools.base import ToolException

from app.embeddings.base_embedding_store import EmbeddingStore
from app.embeddings.embedding_services import EmbeddingModel
from app.log import req_id_cv
from app.metrics.tg_proxy import TigerGraphConnectionProxy
from app.py_schemas import GenerateFunctionResponse, MapQuestionToSchemaResponse
from app.tools.logwriter import LogWriter

from .validation_utils import (
    InvalidFunctionCallException,
    MapQuestionToSchemaException,
    NoDocumentsFoundException,
    validate_function_call,
    validate_schema,
)

logger = logging.getLogger(__name__)


class GenerateFunction(BaseTool):
    """GenerateFunction Tool.
    为问题生成和执行 适当函数调用 的工具。
    """

    name = "GenerateFunction"
    description = "Generates and executes a function call on the database. Always use MapQuestionToSchema before this tool."
    conn: TigerGraphConnectionProxy = None
    llm: LLM = None
    prompt: str = None
    handle_tool_error: bool = True
    embedding_model: EmbeddingModel = None
    embedding_store: EmbeddingStore = None
    args_schema: Type[MapQuestionToSchemaResponse] = MapQuestionToSchemaResponse

    def __init__(self, conn, llm, prompt, embedding_model, embedding_store):
        """Initialize GenerateFunction.
        Args:
            conn (TigerGraphConnection):
                pyTigerGraph TigerGraphConnection connection to the appropriate database/graph with correct permissions
            llm (LLM_Model):
                LLM_Model class to interact with an external LLM API.
            prompt (str):
                prompt to use with the LLM_Model. Varies depending on LLM service.
            embedding_model (EmbeddingModel):
                The model used to generate embeddings for function retrieval.
            embedding_store (EmbeddingStore):
                The embedding store to retrieve functions from.
        """
        super().__init__()
        logger.debug(f"request_id={req_id_cv.get()} GenerateFunction instantiated")
        self.conn = conn
        self.llm = llm
        self.prompt = prompt
        self.embedding_model = embedding_model
        self.embedding_store = embedding_store

    def _run(
        self,
        question: str,
        target_vertex_types: List[str] = [],
        target_vertex_attributes: Dict[str, List[str]] = {},
        target_vertex_ids: Dict[str, List[str]] = {},
        target_edge_types: List[str] = [],
        target_edge_attributes: Dict[str, List[str]] = {},
    ) -> str:
        """Run the tool.
        Args:
            question (str):
                The question to answer with the database.
            target_vertex_types (List[str]):
                The list of vertex types the question mentions.
            target_vertex_attributes (Dict[str, List[str]]):
                The dictionary of vertex attributes the question mentions, in the form {"vertex_type": ["attr1", "attr2"]}
            target_vertex_ids (Dict[str, List[str]):
                The dictionary of vertex ids the question mentions, in the form of {"vertex_type": ["v_id1", "v_id2"]}
            target_edge_types (List[str]):
                The list of edge types the question mentions.
            target_edge_attributes (Dict[str, List[str]]):
                The dictionary of edge attributes the question mentions, in the form {"edge_type": ["attr1", "attr2"]}
        """
        LogWriter.info(f"request_id={req_id_cv.get()} ENTRY GenerateFunction._run()")

        if target_vertex_types == [] and target_edge_types == []:
            return "No vertex or edge types recognized. MapQuestionToSchema and then try again."

        try:
            validate_schema(
                self.conn,
                target_vertex_types,
                target_edge_types,
                target_vertex_attributes,
                target_edge_attributes,
            )
        except MapQuestionToSchemaException as e:
            LogWriter.warning(
                f"request_id={req_id_cv.get()} WARN input schema not valid"
            )
            return e

        # 根据提供的点边的信息，增强原始问题字符串，用于后续的文档检索。
        lookup_question = question + " "
        if target_vertex_types != []:
            lookup_question += "using vertices: " + str(target_vertex_types) + " "
        if target_edge_types != []:
            lookup_question += "using edges: " + str(target_edge_types)

        logger.debug_pii(
            f"request_id={req_id_cv.get()} retrieving documents for question={lookup_question}"
        )
        # 创建动态输出解析器，用于解析LLM的输出
        func_parser = PydanticOutputParser(pydantic_object=GenerateFunctionResponse)
        # 初始化问题的模板
        PROMPT = PromptTemplate(
            template=self.prompt,
            input_variables=[
                "question",
                "vertex_types",
                "edge_types",
                "vertex_attributes",
                "vertex_ids",
                "edge_attributes",
                "doc1",
                "doc2",
                "doc3",
                "doc4",
                "doc5",
                "doc6"
            ],
            partial_variables={
                "format_instructions": func_parser.get_format_instructions()
            },
        )

        # 在嵌入存储（embedding store）中检索与该问题最相关tok_k个的文档。
        pytg_docs = self.embedding_store.retrieve_similar(
            self.embedding_model.embed_query(lookup_question),
            top_k=3,
            filter_expr="graphname == 'all'"
        )

        custom_docs = self.embedding_store.retrieve_similar(
            self.embedding_model.embed_query(lookup_question),
            top_k=3,
            filter_expr="graphname == '{}'".format(
                self.conn.graphname
            ),
        )

        # Prioritize pyTigerGraph docs over custom docs
        docs = pytg_docs + custom_docs

        valid_function_calls = [x["function_header"] for x in self.embedding_store.list_registered_documents(output_fields=["function_header"])]

        if len(docs) == 0:
            LogWriter.warning(f"request_id={req_id_cv.get()} WARN no documents found")
            raise NoDocumentsFoundException

        inputs = [
            {
                "question": question,
                "vertex_types": target_vertex_types,
                "edge_types": target_edge_types,
                "vertex_attributes": target_vertex_attributes,
                "vertex_ids": target_vertex_ids,
                "edge_attributes": target_edge_attributes,
                "doc1": docs[0].page_content,
                "doc2": docs[1].page_content if len(docs) > 1 else "",
                "doc3": docs[2].page_content if len(docs) > 2 else "",
                "doc4": docs[3].page_content if len(docs) > 3 else "",
                "doc5": docs[4].page_content if len(docs) > 4 else "",
                "doc6": docs[5].page_content if len(docs) > 5 else ""
            }
        ]

        logger.debug(f"request_id={req_id_cv.get()} retrieved documents={docs}")

        chain = LLMChain(llm=self.llm, prompt=PROMPT)
        # 使用 LLMChain，结合提供的提示模板和上面收集的数据，将问题转化为函数调用。
        generated = chain.apply(inputs)[0]["text"]
        logger.debug(f"request_id={req_id_cv.get()} generated function")
        # 解析生成的结果
        generated = func_parser.invoke(generated)
        # LogWriter.info(f"generated_function: {generated}")

        try:
            # 验证，获取调用名
            parsed_func = validate_function_call(
                self.conn, generated.connection_func_call, valid_function_calls
            )
        except InvalidFunctionCallException as e:
            LogWriter.warning(
                f"request_id={req_id_cv.get()} EXIT GenerateFunction._run() with exception={e}"
            )
            return e

        try:
            loc = {}
            exec("res = conn." + parsed_func, {"conn": self.conn}, loc)
            LogWriter.info(f"request_id={req_id_cv.get()} EXIT GenerateFunction._run()")
            # 定义返回格式对象
            return {
                "function_call": parsed_func,
                "result": json.dumps(loc["res"]),
                "reasoning": generated.func_call_reasoning,
            }
            # return "Function {} produced the result {}, due to reason {}".format(generated, json.dumps(loc["res"]), generated.func_call_reasoning)
        except Exception as e:
            LogWriter.warning(
                f"request_id={req_id_cv.get()} EXIT GenerateFunction._run() with exception={e}"
            )
            raise ToolException(
                "The function {} did not execute correctly. Please rephrase your question and try again".format(
                    generated
                )
            )

    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

    # def _handle_error(error:Union[ToolException, MapQuestionToSchemaException]) -> str:
    #    return  "The following errors occurred during tool execution:" + error.args[0]+ "Please make sure the question is mapped to the schema correctly"
