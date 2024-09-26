
import chromadb
from chromadb.config import Settings
import uuid

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())



class CollectionManager:
    def __init__(self, data_path, tenant, database, allow_reset=True, anonymized_telemetry=False):
        self.client = chromadb.PersistentClient(
            path=data_path,
            settings=Settings(allow_reset=allow_reset, anonymized_telemetry=anonymized_telemetry),
            tenant=tenant,
            database=database,
        )

    def get_collection(self, collection_name):
        """Get or create a collection by name."""
        return self.client.get_collection(name=collection_name)

    def delete_collection(self, collection_name):
        """Delete the collection by name."""
        self.client.delete_collection(name=collection_name)

    def switch_database(self, new_database):
        """Switch to a different database."""
        self.client.database = new_database

class DocumentManager:
    def __init__(self, collection):
        self.collection = collection

    def get_documents(self):
        """Retrieve all documents and their metadata from the collection."""
        documents = self.collection.get()
        return documents.get("documents"), documents.get("metadatas")

    def create_batches(self, ids, metadatas, embeddings, documents, batch_size=100):
        """Generator function to create batches of data."""
        for i in range(0, len(ids), batch_size):
            yield {
                "ids": ids[i:i+batch_size],
                "metadatas": metadatas[i:i+batch_size],
                "embeddings": embeddings[i:i+batch_size],
                "documents": documents[i:i+batch_size],
            }

    def add_documents(self, texts, metadatas, embeddings, batch_size=100):
        """Add documents to the collection."""
        # Create new document ids
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add documents in batches
        for batch in self.create_batches(
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
            documents=texts,
            batch_size=batch_size
        ):
            self.collection.add(**batch)

# Example usage:

CHROMA_DATA_PATH = "/path/to/chroma"
CHROMA_TENANT = "tenant_name"
CHROMA_DATABASE = "db1"

# Initialize the collection manager
collection_manager = CollectionManager(
    data_path=CHROMA_DATA_PATH,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

# Get a collection
collection = collection_manager.get_collection(collection_name="my_collection")

# Initialize the document manager
document_manager = DocumentManager(collection=collection)

# Add documents
texts = ["doc1", "doc2", "doc3"]
metadatas = [{"type": "text"}, {"type": "text"}, {"type": "text"}]
embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

document_manager.add_documents(texts=texts, metadatas=metadatas, embeddings=embeddings)

# Switch to another database
collection_manager.switch_database("db2")
