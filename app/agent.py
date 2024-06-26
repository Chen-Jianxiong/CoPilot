import time
from langchain.agents import AgentType, initialize_agent
from typing import List, Union
import logging

from pyTigerGraph import TigerGraphConnection

from app.tools import GenerateFunction, MapQuestionToSchema
from app.embeddings.embedding_services import EmbeddingModel
from app.embeddings.base_embedding_store import EmbeddingStore

from app.metrics.prometheus_metrics import metrics
from app.metrics.tg_proxy import TigerGraphConnectionProxy
from app.llm_services.base_llm import LLM_Model

from app.log import req_id_cv
from app.tools.logwriter import LogWriter

logger = logging.getLogger(__name__)


class TigerGraphAgent:
    """TigerGraph Agent Class

    The TigerGraph Agent Class combines the various dependencies needed for a AI Agent to reason with data in a TigerGraph database.

    Args:
        llm_provider (LLM_Model):
            a LLM_Model class that connects to an external LLM API service.
        db_connection (TigerGraphConnection):
            a PyTigerGraph TigerGraphConnection object instantiated to interact with the desired database/graph and authenticated with correct roles.
        embedding_model (EmbeddingModel):
            a EmbeddingModel class that connects to an external embedding API service.
        embedding_store (EmbeddingStore):
            a EmbeddingStore class that connects to an embedding store to retrieve pyTigerGraph and custom query documentation from.
    """

    def __init__(
        self,
        llm_provider: LLM_Model,
        db_connection: "TigerGraphConnectionProxy",
        embedding_model: EmbeddingModel,
        embedding_store: EmbeddingStore,
    ):
        self.conn = db_connection

        self.llm = llm_provider
        self.model_name = embedding_model.model_name

        # 先将问题文本映射为图schema模式
        self.mq2s = MapQuestionToSchema(
            self.conn, self.llm.model, self.llm.map_question_schema_prompt
        )
        # 再根据问题和图schema模式生成函数调用。
        self.gen_func = GenerateFunction(
            self.conn,
            self.llm.model,
            self.llm.generate_function_prompt,
            embedding_model,
            embedding_store,
        )

        tools = [self.mq2s, self.gen_func]
        logger.debug(f"request_id={req_id_cv.get()} agent tools created")
        # 会在调用agent()时自动顺序执行tools
        self.agent = initialize_agent(
            tools,
            self.llm.model,
            # 类型：结构化 zero shot
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            return_intermediate_steps=True,
            # max_iterations=7,
            early_stopping_method="generate",
            handle_parsing_errors=True,
        )
        logger.debug(f"request_id={req_id_cv.get()} agent initialized")

    def question_for_agent(self, question: str):
        """Question for Agent.

        向代理询问一个由数据库回答的问题。返回代理响应或引发异常。

        Args:
            question (str):
                The question to ask the agent
        """
        start_time = time.time()
        metrics.llm_inprogress_requests.labels(self.model_name).inc()

        try:
            LogWriter.info(f"request_id={req_id_cv.get()} ENTRY question_for_agent")
            logger.debug_pii(
                f"request_id={req_id_cv.get()} question_for_agent question={question}"
            )
            # 调用代理处理问题
            resp = self.agent({"input": question})
            LogWriter.info(f"request_id={req_id_cv.get()} EXIT question_for_agent")
            return resp
        except Exception as e:
            metrics.llm_query_error_total.labels(self.model_name).inc()
            LogWriter.error(f"request_id={req_id_cv.get()} FAILURE question_for_agent")
            import traceback

            traceback.print_exc()
            raise e
        finally:
            metrics.llm_request_total.labels(self.model_name).inc()
            metrics.llm_inprogress_requests.labels(self.model_name).dec()
            duration = time.time() - start_time
            metrics.llm_request_duration_seconds.labels(self.model_name).observe(
                duration
            )
