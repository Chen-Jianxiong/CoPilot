class BaseConceptCreator:
    """
    自动化地从关系数据中推导出新的实体概念并建立丰富的语义链接
    """
    def __init__(self, conn, llm, embedding_service):
        self.conn = conn
        self.llm = llm
        self.embedding_service = embedding_service

    def _install_query(self, query_name):
        """
        安装 gsql 查询语句
        """
        with open(
            f"app/gsql/supportai/concept_curation/concept_creation/{query_name}.gsql",
            "r",
        ) as f:
            query = f.read()
        res = self.conn.gsql(
            "USE GRAPH "
            + self.conn.graphname
            + "\n"
            + query
            + "\n INSTALL QUERY "
            + query_name
        )
        return res

    def _check_query_install(self, query_name):
        endpoints = self.conn.getEndpoints(
            dynamic=True
        )
        installed_queries = [q.split("/")[-1] for q in endpoints]

        if query_name not in installed_queries:
            # 在数据库中安装查询
            return self._install_query(query_name)
        else:
            return True

    def create_concepts(self):
        """
        运行查询语句
        """
        raise NotImplementedError


class RelationshipConceptCreator(BaseConceptCreator):
    def __init__(self, conn, llm, embedding_service):
        super().__init__(conn, llm, embedding_service)
        # 检查查询在数据库中是否存在，若不存在则安装查询
        # 分析和处理图中的 Relationship 顶点，以创建和记录那些出现频率超过指定阈值的关系概念节点。
        self._check_query_install("Build_Relationship_Concepts")

    def create_concepts(self, minimum_cooccurrence=5):
        # 指定了最小共同出现的次数，以确定关系概念的阈值。
        res = self.conn.runInstalledQuery(
            "Build_Relationship_Concepts", {"occurence_min": minimum_cooccurrence}
        )
        return res


class EntityConceptCreator(BaseConceptCreator):
    def __init__(self, conn, llm, embedding_service):
        super().__init__(conn, llm, embedding_service)
        # 创建与现有关系概念相关联的新实体概念，并建立相应的关系和描述。
        self._check_query_install("Build_Entity_Concepts")

    def create_concepts(self):
        res = self.conn.runInstalledQuery("Build_Entity_Concepts")
        return res


class CommunityConceptCreator(BaseConceptCreator):
    """ 社区概念创造 """
    def __init__(self, conn, llm, embedding_service):
        super().__init__(conn, llm, embedding_service)
        # 识别图中的社群（或组件），并基于这些社群创建新的概念顶点。
        # 专门用于处理图中的顶点，将它们按连通组件分组，然后根据组件大小和类型创建相应的概念顶点，并建立相应的关系。
        self._check_query_install("Build_Community_Concepts")

    def create_concepts(self, min_community_size=10, max_community_size=100):
        res = self.conn.runInstalledQuery(
            "Build_Community_Concepts",
            {
                "v_type_set": ["Entity", "Relationship"],
                "e_type_set": ["IS_HEAD_OF", "HAS_TAIL"],
                "min_comm_size": min_community_size,
                "max_comm_size": max_community_size,
            },
        )
        return res


class HigherLevelConceptCreator(BaseConceptCreator):
    """ 高级概念创造 """
    def __init__(self, conn, llm, embedding_service):
        super().__init__(conn, llm, embedding_service)
        self._check_query_install("Build_Community_Concepts")
        # 识别两个概念（c1 和 c2）在图中的共现关系。
        # 这涉及到探索和分析两个概念的关联路径和可能的共同实体或关系，进而判定它们在图中的关系网络结构。
        self._check_query_install("getEntityRelationshipConceptCooccurrence")
        # 通过分析概念之间的共现频率来构建概念树。
        # 主要用于连接共现次数达到指定阈值的概念，形成新的概念节点，并将这些新节点以层级关系添加到现有的概念树中。
        self._check_query_install("Build_Concept_Tree")

    def create_concepts(self, min_community_size=5, max_community_size=100):
        self.conn.runInstalledQuery(
            "Build_Concept_Tree", {"min_cooccurence": min_community_size}
        )
        res = self.conn.runInstalledQuery(
            "Build_Community_Concepts",
            {
                "v_type_set": ["Concept"],
                "e_type_set": [
                    "HAS_RELATIONSHIP",
                    "reverse_HAS_RELATIONSHIP",
                    "IS_CHILD_OF",
                    "reverse_IS_CHILD_OF",
                ],
                "min_comm_size": min_community_size,
                "max_comm_size": max_community_size,
            },
        )
        return res
