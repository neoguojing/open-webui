from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    JSONLoader,
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
import faiss
import os
import glob
from typing import Any,List,Dict,Iterator, Optional, Sequence, Union
import shutil
import requests
import validators
import socket
import chromadb
from chromadb import Settings

import logging
log = logging.getLogger(__name__)
log.setLevel("retriever")


class KnowledgeManager:
    def __init__(self, base_path="./knowledge_bases", embedding_dim=512, batch_size=16):
        self.embedding =OllamaEmbeddings(
            model="bge-m3",
            base_url="http://192.168.1.7:11434",
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_DATA_PATH,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    
    def query_doc_with_hybrid_search(
        self,
        collection_name: str,
        query: str,
        embedding_function,
        k: int,
        reranking_function,
        r: float,
    ):
        try:
            collection = self.client.get_collection(name=collection_name)
            documents = collection.get()  # get all documents

            bm25_retriever = BM25Retriever.from_texts(
                texts=documents.get("documents"),
                metadatas=documents.get("metadatas"),
            )
            bm25_retriever.k = k

            chroma_retriever = ChromaRetriever(
                collection=collection,
                embedding_function=embedding_function,
                top_n=k,
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
            )

            compressor = RerankCompressor(
                embedding_function=embedding_function,
                top_n=k,
                reranking_function=reranking_function,
                r_score=r,
            )

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )

            result = compression_retriever.invoke(query)
            result = {
                "distances": [[d.metadata.get("score") for d in result]],
                "documents": [[d.page_content for d in result]],
                "metadatas": [[d.metadata for d in result]],
            }

            log.info(f"query_doc_with_hybrid_search:result {result}")
            return result
        except Exception as e:
            raise e
    
    def query_doc(self,
        collection_name: str,
        query: str,
        k: int,
    ):
        try:
            collection = self.client.get_collection(name=collection_name)
            query_embeddings = self.embed_query(query)

            result = collection.query(
                query_embeddings=[query_embeddings],
                n_results=k,
            )

            log.info(f"query_doc:result {result}")
            return result
        except Exception as e:
            raise e
    
    def embed_query(self,input: str):
        single_vector =  self.embedding.embed_query(input)
        return  single_vector

    def embed_documents(self,inputs):
        vectors =  self.embedding.embed_documents(inputs)
        return  vectors
    
    def resolve_hostname(self,hostname):
        # Get address information
        addr_info = socket.getaddrinfo(hostname, None)

        # Extract IP addresses from address information
        ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
        ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

        return ipv4_addresses, ipv6_addresses
    
    def validate_url(self,url: Union[str, Sequence[str]]):
        if isinstance(url, str):
            if isinstance(validators.url(url), validators.ValidationError):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
            if not ENABLE_RAG_LOCAL_WEB_FETCH:
                # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
                parsed_url = urllib.parse.urlparse(url)
                # Get IPv4 and IPv6 addresses
                ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
                # Check if any of the resolved addresses are private
                # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
                for ip in ipv4_addresses:
                    if validators.ipv4(ip, private=True):
                        raise ValueError(ERROR_MESSAGES.INVALID_URL)
                for ip in ipv6_addresses:
                    if validators.ipv6(ip, private=True):
                        raise ValueError(ERROR_MESSAGES.INVALID_URL)
            return True
        elif isinstance(url, Sequence):
            return all(validate_url(u) for u in url)
        else:
            return False
        
    def store_docs_in_vector_db(
            self,docs, collection_name, metadata: Optional[dict] = None, overwrite: bool = False
    ) -> bool:
        log.info(f"store_docs_in_vector_db {docs} {collection_name}")

        texts = [doc.page_content for doc in docs]
        metadatas = [{**doc.metadata, **(metadata if metadata else {})} for doc in docs]

        # ChromaDB does not like datetime formats
        # for meta-data so convert them to string.
        for metadata in metadatas:
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = str(value)

        try:
            if overwrite:
                for collection in CHROMA_CLIENT.list_collections():
                    if collection_name == collection.name:
                        log.info(f"deleting existing collection {collection_name}")
                        CHROMA_CLIENT.delete_collection(name=collection_name)

            collection = CHROMA_CLIENT.create_collection(name=collection_name)

            embedding_func = get_embedding_function(
                app.state.config.RAG_EMBEDDING_ENGINE,
                app.state.config.RAG_EMBEDDING_MODEL,
                app.state.sentence_transformer_ef,
                app.state.config.OPENAI_API_KEY,
                app.state.config.OPENAI_API_BASE_URL,
                app.state.config.RAG_EMBEDDING_OPENAI_BATCH_SIZE,
            )

            embedding_texts = list(map(lambda x: x.replace("\n", " "), texts))
            embeddings = embedding_func(embedding_texts)

            for batch in create_batches(
                api=CHROMA_CLIENT,
                ids=[str(uuid.uuid4()) for _ in texts],
                metadatas=metadatas,
                embeddings=embeddings,
                documents=texts,
            ):
                collection.add(*batch)

            return True
        except Exception as e:
            if e.__class__.__name__ == "UniqueConstraintError":
                return True

            log.exception(e)

            return False
    
    def get_web_loader(self,url: Union[str, Sequence[str]], verify_ssl: bool = True):
        # Check if the URL is valid
        if not validate_url(url):
            raise ValueError(ERROR_MESSAGES.INVALID_URL)
        return SafeWebBaseLoader(
            url,
            verify_ssl=verify_ssl,
            requests_per_second=RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
            continue_on_failure=True,
        )
    
    def get_youtube_loader(self,url: Union[str, Sequence[str]]):
        loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=app.state.config.YOUTUBE_LOADER_LANGUAGE,
                translation=app.state.YOUTUBE_LOADER_TRANSLATION,
            )
        return loader

    def get_loader(self,filename: str, file_content_type: str, file_path: str):
        file_ext = filename.split(".")[-1].lower()
        known_type = True

        known_source_ext = [
            "go",
            "py",
            "java",
            "sh",
            "bat",
            "ps1",
            "cmd",
            "js",
            "ts",
            "css",
            "cpp",
            "hpp",
            "h",
            "c",
            "cs",
            "sql",
            "log",
            "ini",
            "pl",
            "pm",
            "r",
            "dart",
            "dockerfile",
            "env",
            "php",
            "hs",
            "hsc",
            "lua",
            "nginxconf",
            "conf",
            "m",
            "mm",
            "plsql",
            "perl",
            "rb",
            "rs",
            "db2",
            "scala",
            "bash",
            "swift",
            "vue",
            "svelte",
            "msg",
            "ex",
            "exs",
            "erl",
            "tsx",
            "jsx",
            "hs",
            "lhs",
        ]

        if (
            app.state.config.CONTENT_EXTRACTION_ENGINE == "tika"
            and app.state.config.TIKA_SERVER_URL
        ):
            if file_ext in known_source_ext or (
                file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TikaLoader(file_path, file_content_type)
        else:
            if file_ext == "pdf":
                loader = PyPDFLoader(
                    file_path, extract_images=app.state.config.PDF_EXTRACT_IMAGES
                )
            elif file_ext == "csv":
                loader = CSVLoader(file_path)
            elif file_ext == "rst":
                loader = UnstructuredRSTLoader(file_path, mode="elements")
            elif file_ext == "xml":
                loader = UnstructuredXMLLoader(file_path)
            elif file_ext in ["htm", "html"]:
                loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
            elif file_ext == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_content_type == "application/epub+zip":
                loader = UnstructuredEPubLoader(file_path)
            elif (
                file_content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or file_ext in ["doc", "docx"]
            ):
                loader = Docx2txtLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ] or file_ext in ["xls", "xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ] or file_ext in ["ppt", "pptx"]:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_ext == "msg":
                loader = OutlookMessageLoader(file_path)
            elif file_ext ==".json":
                loader = JSONLoader(file_path)
            elif file_ext in known_source_ext or (
                file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TextLoader(file_path, autodetect_encoding=True)
                known_type = False

        return loader, known_type

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


knowledgeBase = KnowledgeManager()


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(zip(documents, scores.tolist()))
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results


class ChromaRetriever(BaseRetriever):
    collection: Any
    embedding_function: Any
    top_n: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_embeddings = self.embedding_function(query)

        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=self.top_n,
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results

class TikaLoader:
    def __init__(self, file_path, mime_type=None):
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            data = f.read()

        if self.mime_type is not None:
            headers = {"Content-Type": self.mime_type}
        else:
            headers = {}

        endpoint = app.state.config.TIKA_SERVER_URL
        if not endpoint.endswith("/"):
            endpoint += "/"
        endpoint += "tika/text"

        r = requests.put(endpoint, data=data, headers=headers)

        if r.ok:
            raw_metadata = r.json()
            text = raw_metadata.get("X-TIKA:content", "<No text content found>")

            if "Content-Type" in raw_metadata:
                headers["Content-Type"] = raw_metadata["Content-Type"]

            log.info("Tika extracted text: %s", text)

            return [Document(page_content=text, metadata=headers)]
        else:
            raise Exception(f"Error calling Tika: {r.reason}")
        
class SafeWebBaseLoader(WebBaseLoader):
    """WebBaseLoader with enhanced error handling for URLs."""

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path with error handling."""
        for path in self.web_paths:
            try:
                soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
                text = soup.get_text(**self.bs_get_text_kwargs)

                # Build metadata
                metadata = {"source": path}
                if title := soup.find("title"):
                    metadata["title"] = title.get_text()
                if description := soup.find("meta", attrs={"name": "description"}):
                    metadata["description"] = description.get(
                        "content", "No description found."
                    )
                if html := soup.find("html"):
                    metadata["language"] = html.get("lang", "No language found.")

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error loading {path}: {e}")
