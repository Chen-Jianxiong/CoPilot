from app.supportai.retrievers import BaseRetriever
from app.metrics.tg_proxy import TigerGraphConnectionProxy


class HNSWOverlapRetriever(BaseRetriever):
    """
    实现一个基于 HNSW 索引的 overlap 搜索。
    """
    def __init__(
        self,
        embedding_service,
        embedding_store,
        llm_service,
        connection: TigerGraphConnectionProxy,
    ):
        super().__init__(embedding_service, embedding_store, llm_service, connection)
        # 检查查询在数据库中是否存在，若不存在则安装查询
        # 子查询函数，主要用于执行基于 Milvus 的高效向量搜索，找出与给定查询向量最相似的顶点集合。
        self._check_query_install("HNSW_Search_Sub")
        # 从一个向量搜索系统中检索与给定查询向量最相似的顶点，并进一步探索这些顶点的关系和属性。
        self._check_query_install("HNSW_Overlap_Search")

    def search(self, question, indices, top_k=1, num_hops=2, num_seen_min=1):
        # 生成嵌入向量
        query_embedding = self._generate_embedding(question)

        # 执行 HNSW_Overlap_Search 搜索查询
        res = self.conn.runInstalledQuery(
            "HNSW_Overlap_Search",
            # 添加存储库的参数到查询中
            self.embedding_store.add_connection_parameters(
                {
                    "query_vector_as_string": query_embedding,
                    "v_types": indices,
                    "collection_prefix": self.conn.graphname,
                    "top_k": top_k,
                    "num_hops": num_hops,
                    "num_seen_min": num_seen_min,
                }
            ),
        )
        return res

    def retrieve_answer(self, question, index, top_k=1, num_hops=2, num_seen_min=1):
        # 执行搜索获取结果
        retrieved = self.search(question, index, top_k, num_hops, num_seen_min)
        # llm回答并生成响应
        return self._generate_response(question, retrieved)
