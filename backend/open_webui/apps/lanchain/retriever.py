from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import EnsembleRetriever
import faiss
import os
import glob
from typing import Any,List,Dict
import shutil


class KnowledgeBaseManager:
    def __init__(self, base_path="./knowledge_bases", embedding_dim=512, batch_size=16):
        self.base_path = base_path
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embeddings = None
        self.knowledge_bases: Dict[str, FAISS] = {}
        os.makedirs(self.base_path, exist_ok=True)

        faiss_files = glob.glob(os.path.join(base_path, '*.faiss'),recursive=True)
        # 获取不带后缀的名称
        # file_names_without_extension = [os.path.splitext(os.path.basename(file))[0] for file in faiss_files]
        # 获取 .faiss 文件的上级目录的名称
        db_names = [os.path.basename(os.path.dirname(file)) for file in faiss_files]

        for name in db_names:
            self.load_knowledge_base(name)


    def create_knowledge_base(self, name: str):
        index = faiss.IndexFlatL2(self.embedding_dim)
        index = FAISS(self.embeddings, index, InMemoryDocstore(), {})
        if name in self.knowledge_bases:
            print(f"Knowledge base '{name}' already exists.")
            return
        
        self.knowledge_bases[name] = index
        self.save_knowledge_base(name)
        print(f"Knowledge base '{name}' created.")

    def delete_knowledge_base(self, name: str):
        if name in self.knowledge_bases and name != "":
            del self.knowledge_bases[name]
            index_path = os.path.join(self.base_path, name)
            shutil.rmtree(index_path)
            print(f"Knowledge base '{name}' deleted.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    def load_knowledge_base(self, name: str):
        index_path = os.path.join(self.base_path, name)
        if os.path.exists(index_path):
            self.knowledge_bases[name] = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Knowledge base '{name}' loaded.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    def save_knowledge_base(self, name: str):
        if name in self.knowledge_bases:
            index_path = os.path.join(self.base_path, name)
            os.makedirs(index_path, exist_ok=True)
            self.knowledge_bases[name].save_local(index_path)
            print(f"Knowledge base '{name}' saved.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    # Document(page_content = '渠道版', metadata = {
	# 'source': './files/input/PS004.pdf',
	# 'page': 0
    # }), Document(page_content = '2/20.', metadata = {
    #     'source': './files/input/PS004.pdf',
    #     'page': 1
    # })
    def add_documents_to_knowledge_base(self, name: str, file_paths: List[str]):
        if name not in self.knowledge_bases:
            print(f"Knowledge base '{name}' does not exist.")
            self.create_knowledge_base(name)
        
        index = self.knowledge_bases[name]
        documents = self.load_documents(file_paths)
        print(f"Loaded {len(documents)} documents.")
        print(documents)
        pages = self.split_documents(documents)
        print(f"Split documents into {len(pages)} pages.")
        # print(pages)
        
        doc_ids = []
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i:i+self.batch_size]
            doc_ids.extend(index.add_documents(batch))
        
        self.save_knowledge_base(name)
        return doc_ids

    def load_documents(self, file_paths: List[str]):
        documents = []
        for file_path in file_paths:
            loader = self.get_loader(file_path)
            documents.extend(loader.load())
        return documents

    def get_loader(self, file_path: str):
        if file_path.endswith('.txt'):
            return TextLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path)
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(separators=[
                                                    "\n\n",
                                                    "\n",
                                                    " ",
                                                    ".",
                                                    ",",
                                                    "\u200b",  # Zero-width space
                                                    "\uff0c",  # Fullwidth comma
                                                    "\u3001",  # Ideographic comma
                                                    "\uff0e",  # Fullwidth full stop
                                                    "\u3002",  # Ideographic full stop
                                                    "",
                                                ],
                                                chunk_size=512, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def retrieve_documents(self, names: List[str], query: str):
        results = []
        for name in names:
            if name not in self.knowledge_bases:
                print(f"Knowledge base '{name}' does not exist.")
                continue
            
            retriever = self.knowledge_bases[name].as_retriever(
                search_type="mmr",
                search_kwargs={"score_threshold": 0.5, "k": 3}
            )
            docs = retriever.get_relevant_documents(query)
            results.extend([{"name": name, "content": doc.page_content,"meta": doc.metadata} for doc in docs])
            
        
        return results
    
    def get_retriever(self,names: List[str]):
        retrievers = []
        weights = []
        for name in names:
            if name not in self.knowledge_bases:
                print(f"Knowledge base '{name}' does not exist.")
                continue

            weights.append(0.5)
            retrievers.append(self.knowledge_bases[name].as_retriever(
                    search_type="mmr",
                    search_kwargs={"score_threshold": 0.5, "k": 3}
                )
            )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers, weights=weights
        )
        return ensemble_retriever
    
    def get_bases(self):
        data = self.knowledge_bases.keys()
        return list(data)
    
    def get_df_bases(self):
        import pandas as pd
        data = self.knowledge_bases.keys()
        return pd.DataFrame(list(data), columns=['列表'])

knowledgeBase = KnowledgeBaseManager()
