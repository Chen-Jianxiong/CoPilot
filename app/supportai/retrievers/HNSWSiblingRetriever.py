from app.metrics.tg_proxy import TigerGraphConnectionProxy
from app.supportai.retrievers import BaseRetriever


class HNSWSiblingRetriever(BaseRetriever):
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
        # 通过 HNSW 算法在给定的向量空间中执行高效的近邻搜索，探索相关的内容块，并跟踪和展示顶点之间的关系及内容。
        self._check_query_install("HNSW_Chunk_Sibling_Search")

    def search(self, question, index, top_k=1, lookback=3, lookahead=3, withHyDE=False):
        if withHyDE:
            query_embedding = self._hyde_embedding(question)
        else:
            query_embedding = self._generate_embedding(question)
        res = self.conn.runInstalledQuery(
            "HNSW_Chunk_Sibling_Search",
            self.embedding_store.add_connection_parameters(
                {
                    "v_type": index,
                    "query_vector_as_string": query_embedding,
                    "collection_name": self.conn.graphname + "_" + index,
                    "lookback": lookback,
                    "lookahead": lookahead,
                    "top_k": top_k,
                }
            ),
        )
        return res

    def retrieve_answer(
        self, question, index, top_k=1, lookback=3, lookahead=3, withHyDE=False
    ):
        retrieved = self.search(question, index, top_k, lookback, lookahead, withHyDE)
        return self._generate_response(question, retrieved)
