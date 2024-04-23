from app.supportai.retrievers import BaseRetriever
from app.metrics.tg_proxy import TigerGraphConnectionProxy


class HNSWRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_service,
        embedding_store,
        llm_service,
        connection: TigerGraphConnectionProxy,
    ):
        super().__init__(embedding_service, embedding_store, llm_service, connection)
        # 子查询函数，主要用于执行基于 Milvus 的高效向量搜索，找出与给定查询向量最相似的顶点集合。
        self._check_query_install("HNSW_Search_Sub")
        # 利用 Milvus 进行高效的向量搜索，从而检索与给定查询向量相似的顶点，接着进一步提取和返回这些顶点相关联的内容或定义信息。
        self._check_query_install("HNSW_Search_Content")

    def search(self, question, index, top_k=1, withHyDE=False):
        if withHyDE:
            #  使用HyDE嵌入回答问题，返回嵌入结果
            query_embedding = self._hyde_embedding(question)
        else:
            query_embedding = self._generate_embedding(question)
        # 为params增加 milvus_host、milvus_port
        params = self.embedding_store.add_connection_parameters(
            {
                "v_type": index,
                "query_vector_as_string": query_embedding,
                "collection_name": self.conn.graphname + "_" + index,
                "top_k": top_k,
            }
        )
        # 执行查询
        res = self.conn.runInstalledQuery("HNSW_Search_Content", params)
        return res

    def retrieve_answer(self, question, index, top_k=1, withHyDE=False):
        retrieved = self.search(question, index, top_k, withHyDE)
        return self._generate_response(question, retrieved)
