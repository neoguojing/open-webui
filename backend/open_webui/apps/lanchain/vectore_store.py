import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

class CollectionManager:
    def __init__(self, data_path, tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE, allow_reset=True, anonymized_telemetry=False):
        self.client = chromadb.PersistentClient(
            path=data_path,
            settings=Settings(allow_reset=allow_reset, anonymized_telemetry=anonymized_telemetry),
            database=database
        )
        self.embedding = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")

    def get_or_create_collection(self, collection_name):
        """Get or create a collection by name."""
        try:
            return self.client.get_collection(name=collection_name)
        except chromadb.errors.CollectionAlreadyExistsError:
            return self.create_collection(collection_name)

    def delete_collection(self, collection_name):
        """Delete the collection by name."""
        self.client.delete_collection(name=collection_name)

    def create_collection(self, collection_name):
        """Create a new collection."""
        return self.client.create_collection(name=collection_name)

    def list_collections(self, limit=None, offset=None):
        """List collections with optional pagination."""
        return self.client.list_collections(limit, offset)
    
    def get_vector_store(self, collection_name) -> Chroma:
        """Get or create a vector store for the given collection name."""
        try:
            collection = self.get_or_create_collection(collection_name)
        except chromadb.errors.CollectionAlreadyExistsError as e:
            raise ValueError(f"Collection '{collection_name}' already exists with errors: {e}")
        
        return Chroma(persist_directory=self.client.path, 
                      embedding_function=self.embedding, 
                      collection_name=collection.name)

    def get_documents(self, collection_name) -> list[Document]:
        """Retrieve all documents and their metadata from the collection."""
        collection = self.get_or_create_collection(collection_name)
        result = collection.get()
        
        return [Document(page_content=document['text'], metadata=metadata) 
                for document, metadata in zip(result['documents'], result['metadatas'])]
