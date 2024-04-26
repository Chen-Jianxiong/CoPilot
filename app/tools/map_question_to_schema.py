from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.tools.base import ToolException
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from app.metrics.tg_proxy import TigerGraphConnectionProxy
from app.py_schemas import MapQuestionToSchemaResponse, MapAttributeToAttributeResponse
from typing import List, Dict
from .validation_utils import validate_schema, MapQuestionToSchemaException
import re
import logging
from app.log import req_id_cv
from app.tools.logwriter import LogWriter

logger = logging.getLogger(__name__)


class MapQuestionToSchema(BaseTool):
    """MapQuestionToSchema Tool.
    将问题映射到数据库中的数据类型的工具。应该在GenerateFunction之前执行。
    """

    name = "MapQuestionToSchema"
    description = "Always run first to map the query to the graph's schema. GenerateFunction before using MapQuestionToSchema"
    conn: "TigerGraphConnectionProxy" = None
    llm: LLM = None
    prompt: str = None
    handle_tool_error: bool = True

    def __init__(self, conn, llm, prompt):
        """Initialize MapQuestionToSchema.
        Args:
            conn (TigerGraphConnectionProxy):
                pyTigerGraph TigerGraphConnection connection to the database; this is a proxy which includes metrics gathering.
            llm (LLM_Model):
                LLM_Model class to interact with an external LLM API.
            prompt (str):
                prompt to use with the LLM_Model. Varies depending on LLM service.
        """
        super().__init__()
        logger.debug(f"request_id={req_id_cv.get()} MapQuestionToSchema instantiated")
        self.conn = conn
        self.llm = llm
        self.prompt = prompt

    def _run(self, query: str) -> str:
        """Run the tool.
        Args:
            query (str):
                The user's question.
        """
        LogWriter.info(f"request_id={req_id_cv.get()} ENTRY MapQuestionToSchema._run()")
        # 创建动态输出解析器，用于解析LLM的输出
        parser = PydanticOutputParser(pydantic_object=MapQuestionToSchemaResponse)
        # 初始化重述问题的模板
        RESTATE_QUESTION_PROMPT = PromptTemplate(
            template=self.prompt,
            input_variables=[
                "question",
                "vertices",
                "verticesAttrs",
                "edges",
                "edgesInfo",
            ],
            # format_instructions: MapQuestionToSchemaResponse的格式
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 应用重述链
        restate_chain = LLMChain(llm=self.llm, prompt=RESTATE_QUESTION_PROMPT)

        # 获取图数据库信息
        vertices = self.conn.getVertexTypes()
        edges = self.conn.getEdgeTypes()

        vertices_info = []
        for vertex in vertices:
            vertex_attrs = self.conn.getVertexAttrs(vertex)
            attributes = [attr[0] for attr in vertex_attrs]
            vertex_info = {"vertex": vertex, "attributes": attributes}
            vertices_info.append(vertex_info)

        edges_info = []
        for edge in edges:
            source_vertex = self.conn.getEdgeSourceVertexType(edge)
            target_vertex = self.conn.getEdgeTargetVertexType(edge)
            edge_info = {"edge": edge, "source": source_vertex, "target": target_vertex}
            edges_info.append(edge_info)

        # 使用 LLMChain，结合提供的提示模板和上面收集的数据，生成一个重新表述的问题。
        restate_q = restate_chain.apply(
            [
                {
                    "vertices": vertices,
                    "verticesAttrs": vertices_info,
                    "edges": edges,
                    "edgesInfo": edges_info,
                    "question": query,
                }
            ]
        )[0]["text"]

        logger.debug(f"request_id={req_id_cv.get()} MapQuestionToSchema applied")

        # 解析重述的问题：通过之前创建的 parser 解析重述问题的文本。
        parsed_q = parser.invoke(restate_q)

        logger.debug_pii(
            f"request_id={req_id_cv.get()} MapQuestionToSchema parsed for question={query} into normalized_form={parsed_q}"
        )

        attr_prompt = """For the following source attributes: {parsed_attrs}, map them to the corresponding output attribute in this list: {real_attrs}.
                         Format the response way explained below:
                        {format_instructions}"""

        # 创建属性映射的模板：为属性映射过程定义一个新的提示模板。
        attr_parser = PydanticOutputParser(
            pydantic_object=MapAttributeToAttributeResponse
        )

        ATTR_MAP_PROMPT = PromptTemplate(
            template=attr_prompt,
            input_variables=["parsed_attrs", "real_attrs"],
            partial_variables={
                "format_instructions": attr_parser.get_format_instructions()
            },
        )

        # 再映射一遍属性，使其能够与图数据库的属性名对齐，防止属性名错误
        attr_map_chain = LLMChain(llm=self.llm, prompt=ATTR_MAP_PROMPT)
        # {'vertex_type_1': ['vertex_attribute_1', 'vertex_attribute_2']}
        for vertex in parsed_q.target_vertex_attributes.keys():
            map_attr = attr_map_chain.apply(
                [
                    {
                        "parsed_attrs": parsed_q.target_vertex_attributes[vertex],
                        "real_attrs": [attr[0] for attr in self.conn.getVertexAttrs(vertex)],
                    }
                ]
            )[0]["text"]
            parsed_map = attr_parser.invoke(map_attr).attr_map
            parsed_q.target_vertex_attributes[vertex] = [
                parsed_map[x] for x in list(parsed_q.target_vertex_attributes[vertex])
            ]

        logger.debug(f"request_id={req_id_cv.get()} MapVertexAttributes applied")

        for edge in parsed_q.target_edge_attributes.keys():
            map_attr = attr_map_chain.apply(
                [
                    {
                        "parsed_attrs": parsed_q.target_edge_attributes[edge],
                        "real_attrs": self.conn.getEdgeAttrs(edge),
                    }
                ]
            )[0]["text"]
            parsed_map = attr_parser.invoke(map_attr).attr_map
            parsed_q.target_edge_attributes[edge] = [
                parsed_map[x] for x in list(parsed_q.target_edge_attributes[edge])
            ]

        logger.debug(f"request_id={req_id_cv.get()} MapEdgeAttributes applied")

        try:
            # 验证映射后的架构
            validate_schema(
                self.conn,
                parsed_q.target_vertex_types,
                parsed_q.target_edge_types,
                parsed_q.target_vertex_attributes,
                parsed_q.target_edge_attributes,
            )
        except MapQuestionToSchemaException as e:
            LogWriter.warning(
                f"request_id={req_id_cv.get()} WARN MapQuestionToSchema to validate schema"
            )
            raise e
        LogWriter.info(f"request_id={req_id_cv.get()} EXIT MapQuestionToSchema._run()")
        return parsed_q

    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

    # def _handle_error(self, error:MapQuestionToSchemaException) -> str:
    #    return  "The following errors occurred during tool execution:" + error.args[0]+ "Please make sure to map the question to the schema"
