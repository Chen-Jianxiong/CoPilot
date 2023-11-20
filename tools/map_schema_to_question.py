from langchain.tools import BaseTool
from langchain.llms.base import LLM
from langchain.tools.base import ToolException
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pyTigerGraph import TigerGraphConnection
from langchain.pydantic_v1 import BaseModel, Field, validator
from schemas import MapQuestionToSchemaResponse
from typing import List, Dict
from .validate_against_schema import validate_schema, MapQuestionToSchemaException
import re



class MapQuestionToSchema(BaseTool):
    name = "MapQuestionToSchema"
    description = "Always run first to map the query to the graph's schema. NEVER EXECUTE GenerateFunction before using MapQuestionToSchema"
    conn: "TigerGraphConnection" = None
    llm: LLM = None
    prompt: str = None
    handle_tool_error: bool = True
    
    def __init__(self, conn, llm, prompt):
        super().__init__()
        self.conn = conn
        self.llm = llm
        self.prompt = prompt
        
    def _run(self, query: str) -> str:
        """Use the tool."""

        parser = PydanticOutputParser(pydantic_object=MapQuestionToSchemaResponse)

        RESTATE_QUESTION_PROMPT = PromptTemplate(
            template=self.prompt,
            input_variables=["question", "vertices", "edges"],
            partial_variables = {"format_instructions": parser.get_format_instructions()}
        )
        restate_chain = LLMChain(llm=self.llm, prompt=RESTATE_QUESTION_PROMPT)
        
        restate_q = restate_chain.apply([{"vertices": self.conn.getVertexTypes(),
                                          "question": query,
                                          "edges": self.conn.getEdgeTypes()}])[0]["text"]

        parsed_q = parser.invoke(restate_q)


        try:
            validate_schema(self.conn,
                            parsed_q.target_vertex_types, 
                            parsed_q.target_edge_types,
                            parsed_q.target_vertex_attributes, 
                            parsed_q.target_edge_attributes)
        except MapQuestionToSchemaException as e:
            raise e
    
        return parsed_q
    
    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
        
    #def _handle_error(self, error:MapQuestionToSchemaException) -> str:
    #    return  "The following errors occurred during tool execution:" + error.args[0]+ "Please make sure to map the question to the schema"