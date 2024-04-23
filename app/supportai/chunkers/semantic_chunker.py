from app.supportai.chunkers.base_chunker import BaseChunker
from app.embeddings.embedding_services import EmbeddingModel
from langchain_experimental.text_splitter import (
    SemanticChunker as LangChainSemanticChunker,
)


class SemanticChunker(BaseChunker):
    """ 语义分割器 """
    def __init__(
        self,
        # 嵌入服务，用于计算文本的向量表示
        embedding_serivce: EmbeddingModel,
        # 分割文本的阈值类型
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 0.95,
    ):
        self.emb_model = embedding_serivce
        self.btt = breakpoint_threshold_type
        self.bta = breakpoint_threshold_amount

    def chunk(self, input_string):
        """ 分割文本 """
        text_splitter = LangChainSemanticChunker(
            self.emb_model.embeddings,
            breakpoint_threshold_type=self.btt,
            breakpoint_threshold_amount=self.bta,
        )

        chunks = text_splitter.create_documents([input_string])

        return [x.page_content for x in chunks]

    def __call__(self, input_string):
        return self.chunk(input_string)
