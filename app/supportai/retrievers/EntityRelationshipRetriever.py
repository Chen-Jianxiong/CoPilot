from app.supportai.retrievers import BaseRetriever
from app.supportai.extractors import LLMEntityRelationshipExtractor
from app.metrics.tg_proxy import TigerGraphConnectionProxy


class EntityRelationshipRetriever(BaseRetriever):
    """ 实体关系检索器 """
    def __init__(
        self,
        embedding_service,
        embedding_store,
        llm_service,
        connection: TigerGraphConnectionProxy,
    ):
        super().__init__(embedding_service, embedding_store, llm_service, connection)
        # 根据指定的实体和关系集合，从文档中检索与之相关的内容，并根据相关性得分返回top_k文档片段。
        self._check_query_install("Entity_Relationship_Retrieval")
        self.extractor = LLMEntityRelationshipExtractor(llm_service)

    def search(self, question, top_k=1):
        # 利用llm从提问中提取实体关系
        nodes_rels = self.extractor.document_er_extraction(question)
        # 由tg执行查询语句检索top_K
        res = self.conn.runInstalledQuery(
            "Entity_Relationship_Retrieval",
            {
                "entities": [x["id"] for x in nodes_rels["nodes"]],
                "relationships": [x["type"] for x in nodes_rels["rels"]],
                "top_k": top_k,
            },
        )

        return res

    def retrieve_answer(self, question, top_k=1):
        retrieved = self.search(question, top_k)
        return self._generate_response(question, retrieved)
