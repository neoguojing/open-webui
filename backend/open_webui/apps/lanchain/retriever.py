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
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
import faiss
import os
import glob
from typing import Any,List,Dict,Iterator, Optional, Sequence, Union
import shutil
import requests
import validators
import socket
import chromadb
import urllib.parse
from chromadb import Settings
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter,LLMListwiseRerank,LLMChainFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from .vectore_store import CollectionManager
import logging
log = logging.getLogger(__name__)
log.setLevel("retriever")

from enum import Enum

class SourceType(Enum):
    YOUTUBE = "youtube"
    WEB = "web"
    FILE = "file"

class FilterType(Enum):
    LLM_FILTER = "llm_chain_filter"
    LLM_RERANK = "llm_listwise_rerank"
    RELEVANT_FILTER = "embeddings_filter"

class KnowledgeManager:
    def __init__(self, data_path, tenant, database):
        self.embedding =OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434",
        )

        self.llm =ChatOpenAI(
            model="llama3.1-local",
            openai_api_key="121212",
            base_url="http://localhost:11434/v1",
        )

        self.collection_manager = CollectionManager(data_path,tenant,database)

    def store(self,collection_name: str, source: Union[str, List[str]], source_type: SourceType,
               file_info: Optional[Dict[str, str]] = None):
        """
        存储 URL 或文件，支持单个或多个 source。
        
        参数:
        - collection_name: 存储的集合名称
        - source: 如果是 URL，传入字符串或列表；如果是文件，传入文件路径或文件路径列表
        - source_type: 数据源类型，支持 SourceType.YOUTUBE, SourceType.WEB, SourceType.FILE
        - file_info: 一个包含文件相关信息的字典，只有当 source_type 是 SourceType.FILE 时使用。
                    字典中可以包含:
                    - "file_name": 文件名
                    - "content_type": 文件的内容类型
        """
         # 处理单个或多个 source
        if isinstance(source, str):
            sources = [source]  # 如果是字符串，转为列表
        elif isinstance(source, list):
            sources = source  # 如果是列表，直接使用
        else:
            raise ValueError("Source must be a string or a list of strings.")
    
        loader = None
        try:
            if source_type == SourceType.YOUTUBE:
                loader = self.get_youtube_loader(sources)
            elif source_type == SourceType.WEB:
                loader = self.get_web_loader(sources)
            else:
                if file_info is None:
                    raise ValueError("File information is required for file storage.")
                
                file_name = file_info.get("file_name", "")
                content_type = file_info.get("content_type", "application/octet-stream")

                loader = self.get_loader(file_name,file_path,content_type)
            
            file_path = loader.load()
            docs = self.split_documents(file_path)
            self.collection_manager.get_or_create_vector_store(collection_name).add_documents(docs)
            return True
        except Exception as e:
            if e.__class__.__name__ == "UniqueConstraintError":
                return True
            log.exception(e)
            return False

    def get_compress_retriever(self,retriever,filter_type:FilterType = FilterType.LLM_FILTER):
        relevant_filter = None
        if filter_type == FilterType.LLM_FILTER:
            relevant_filter = LLMChainFilter.from_llm(self.llm)
        
        elif filter_type == FilterType.LLM_RERANK:
            relevant_filter = LLMListwiseRerank.from_llm(self.llm, top_n=1)
        
        elif filter_type == FilterType.RELEVANT_FILTER:
            relevant_filter = EmbeddingsFilter(embeddings=self.embedding, similarity_threshold=0.76)

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding)
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, relevant_filter]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

        return compression_retriever
    
    def get_retriever(self,collection_name,k: int,bm25: bool):
        
        chroma_retriever = self.collection_manager.get_or_create_vector_store(collection_name).as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'lambda_mult': 0.25}
        )
        retriever = chroma_retriever

        if bm25:
            docs = self.collection_manager.get_documents(collection_name)
            bm25_retriever = BM25Retriever.from_documents(documents=docs)
            bm25_retriever.k = k
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
            )
        return retriever
    
    
    def query_doc(self,
        collection_name: Union[str, List[str]],
        query: str,
        k: int,
        bm25: bool = True,
        rerank: bool = True
    ):
        collection_names = []
        if isinstance(collection_name, str):
            collection_names = [collection_name]  # 如果是字符串，转为列表
        elif isinstance(collection_name, list):
            collection_names = collection_name  # 如果是列表，直接使用
        else:
            raise ValueError("Source must be a string or a list of strings.")
        
        try:
            for name in collection_names:
                retriever = self.get_retriever(name,k,bm25=bm25)
                if rerank:
                    retriever = self.get_compress_retriever(retriever)
                docs = retriever.invoke(query)
                return docs
        except Exception as e:
            raise e
    
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
                raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
            if not ENABLE_RAG_LOCAL_WEB_FETCH:
                # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
                parsed_url = urllib.parse.urlparse(url)
                # Get IPv4 and IPv6 addresses
                ipv4_addresses, ipv6_addresses = self.resolve_hostname(parsed_url.hostname)
                # Check if any of the resolved addresses are private
                # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
                for ip in ipv4_addresses:
                    if validators.ipv4(ip, private=True):
                        raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
                for ip in ipv6_addresses:
                    if validators.ipv6(ip, private=True):
                        raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
            return True
        elif isinstance(url, Sequence):
            return all(self.validate_url(u) for u in url)
        else:
            return False
        
    def get_web_loader(self,url: Union[str, Sequence[str]], verify_ssl: bool = True):
        # Check if the URL is valid
        if not self.validate_url(url):
            raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
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

    def get_loader(self,filename: str, file_path: str, file_content_type: str=None):
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

    def split_documents(self, documents,chunk_size=512,chunk_overlap=512):
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
                                                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)


knowledgeBase = KnowledgeManager()


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
