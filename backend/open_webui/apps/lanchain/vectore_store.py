
import chromadb
from chromadb.config import Settings
import uuid
from chromadb.utils.batch_utils import create_batches
from langchain_core.documents import Document
from datetime import datetime
from typing import Iterator, Optional, Sequence, Union
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

class CollectionManager:
    def __init__(self, data_path, tenant=chromadb.DEFAULT_TENANT, 
                 database=chromadb.DEFAULT_DATABASE, allow_reset=True, anonymized_telemetry=False):
        self.client = chromadb.PersistentClient(
            path=data_path,
            settings=Settings(allow_reset=allow_reset, anonymized_telemetry=anonymized_telemetry),
            tenant=tenant,
            database=database,
        )

        self.embedding =OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434",
        )

    def get_collection(self, collection_name):
        """Get or create a collection by name."""
        return self.client.get_collection(name=collection_name)

    def delete_collection(self, collection_name):
        """Delete the collection by name."""
        self.client.delete_collection(name=collection_name)

    def create_collection(self, collection_name):
        """Switch to a different database."""
        self.client.create_collection(name=collection_name)

    def list_collections(self,limit: Optional[int] = None,offset: Optional[int] = None):
        return self.client.list_collections(limit,offset)
    
    def get_collection(self, collection_name):
        """Switch to a different database."""
        self.client.create_collection(name=collection_name)
    
    def get_or_create_vector_store(self,collection_name) -> Chroma:
        return Chroma(collection_name,client=self.client,embedding_function=self.embedding)
    
    def get_documents(self,collection_name) -> list[Document]:
        """Retrieve all documents and their metadata from the collection."""
        collection = self.client.create_collection(name=collection_name)
        documents = collection.get()
        docs = []
        for doc, metadata in zip(documents.get("documents"), documents.get("metadatas")):
            # 组装新的 Document 对象
            d = Document(
                page_content=doc.page_content,  # 从文档中提取 page_content
                metadata=metadata,  # 使用对应的元数据
            )
            docs.append(d)
        return docs
    
    




