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
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from typing import Any,List,Dict,Iterator, Optional, Sequence, Union
import requests
import validators
import socket
import urllib.parse
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter,LLMListwiseRerank,LLMChainFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from vectore_store import CollectionManager
from prompt import DEFAULT_SEARCH_PROMPT
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import AsyncChromiumLoader,AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.retrievers.web_research import QuestionListOutputParser

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from enum import Enum

class SourceType(Enum):
    YOUTUBE = "youtube"
    WEB = "web"
    FILE = "file"

class FilterType(Enum):
    LLM_FILTER = "llm_chain_filter"
    LLM_RERANK = "llm_listwise_rerank"
    RELEVANT_FILTER = "embeddings_filter"
    
# Load HTML
# loader = AsyncChromiumLoader(["https://www.wsj.com"])
# html = loader.load()
# bs_transformer = BeautifulSoupTransformer()

# from langchain.chains import RetrievalQAWithSourcesChain

# user_input = "How do LLM Powered Autonomous Agents work?"
# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm, retriever=web_research_retriever
# )
# result = qa_chain({"question": user_input})
# result

class KnowledgeManager:
    def __init__(self, data_path,ollama_url="http://localhost:11434",model="qwen2.5:14b", tenant=None, database=None):
        self.embedding =OllamaEmbeddings(
            model="bge-m3:latest",
            base_url=ollama_url,
            # num_gpu=100
        )

        self.llm =ChatOpenAI(
            model=model,
            openai_api_key="121212",
            base_url=f"{ollama_url}/v1",
        )
        
        self.search_chain = DEFAULT_SEARCH_PROMPT | self.llm | QuestionListOutputParser()

        self.collection_manager = CollectionManager(data_path)

    def store(self,collection_name: str, source: Union[str, List[str]], source_type: SourceType,
               file_name:str = None,content_type: str = None):
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
                if file_name is None:
                    raise ValueError("File name is required for file storage.")
                loader,known_type = self.get_loader(file_name,source,content_type)
            
            print("start load file---------")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.page_content = doc.page_content.replace("\n", " ")
                doc.page_content = doc.page_content.replace("\t", "")
            print("loader file count:",len(raw_docs))
            docs = self.split_documents(raw_docs)
            print("splited documents count:",docs)
            store = self.collection_manager.get_vector_store(collection_name)
            store.add_documents(docs)
            print("add documents done------")
            return collection_name
        except Exception as e:
            if e.__class__.__name__ == "UniqueConstraintError":
                return True
            log.exception(e)
            return False

    def get_compress_retriever(self,retriever,filter_type:FilterType):
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
        
        chroma_retriever = self.collection_manager.get_vector_store(collection_name).as_retriever(
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
        k: int = 3,
        bm25: bool = False,
        rerank: bool = False,
        filter_type: FilterType = FilterType.LLM_FILTER
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
                    retriever = self.get_compress_retriever(retriever,filter_type)
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
            requests_per_second=10,
            continue_on_failure=True,
        )
        # return AsyncChromiumLoader(url)
    
    def get_youtube_loader(self,url: Union[str, Sequence[str]]):
        loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language='en',
                translation=None
            )
        return loader

    def get_loader(self,filename: str, file_path: str, file_content_type: str=None):
        loader = None
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

        if file_ext == "pdf":
            loader = PyPDFLoader(
                file_path, extract_images=False
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
    # decrease
    def web_parser(self,urls,metadata=None,collection_name=None):
        bs_transformer = BeautifulSoupTransformer()
        vector_store = None
        if collection_name:
            vector_store = self.collection_manager.get_vector_store(collection_name)
        docs = None
        if urls:
            loader = AsyncChromiumLoader(urls)
            log.info("Indexing new urls...")
            docs = loader.load()
            print("load docs:",len(docs))
            docs_transformed = bs_transformer.transform_documents(
                docs, tags_to_extract=["span"]
            )
            docs = list(docs_transformed)
            print("transform docs:",len(docs))
            docs = self.split_documents(docs)
            print("split docs:",len(docs))
            if metadata:
                for doc in docs:
                    doc.metadata=metadata[doc.metadata['source']]
            if vector_store:
                vector_store.add_documents(docs)
        return docs
        
    def web_search(self,query,collection_name="web",region="cn-zh",time="d",max_results=2):
        questions = self.search_chain.invoke({"question":query})
        print("questions:",questions)
        
        search = DuckDuckGoSearchAPIWrapper(region=region, time=time, max_results=max_results,source="news")
        
        urls_to_look = []
        url_meta_map = {}
        try:
            for query in questions:
                print(query)
                search_results = search.results(query,max_results=1)
                log.info("Searching for relevant urls...")
                log.info(f"Search results: {search_results}")
                for res in search_results:
                    print(res)
                    if res.get("link", None):
                        urls_to_look.append(res["link"])
                        # url_meta_map[res["link"]] = res
        
        except Exception as e:
            log.error(f"Error search: {e}")
        
        # print("url_meta_map:",url_meta_map)
        # Relevant urls
        urls = set(urls_to_look)
        
        # docs = self.web_parser(urls,url_meta_map,collection_name)
        return self.store(collection_name,list(urls),SourceType.WEB)
         

    def split_documents(self, documents,chunk_size=1024,chunk_overlap=50):
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


if __name__ == '__main__':
    knowledgeBase = KnowledgeManager(data_path="./test/")
    # knowledgeBase.store(collection_name="test",source="/home/neo/Downloads/ir2023_ashare.docx",
    #                     source_type=SourceType.FILE,file_name='ir2023_ashare.docx')
    docs = knowledgeBase.query_doc("web","中国股市",k=2,bm25=False,rerank=True)
    # print(docs)
    # emb = knowledgeBase.embedding.embed_query("wsewqeqe")
    # print(emb)
    # resp = knowledgeBase.llm.invoke("hhhhh")
    # print(resp)
    # docs = knowledgeBase.web_search("中国的股市如何估值？")
    # docs = knowledgeBase.web_parser(["https://new.qq.com/rain/a/20241005A071AG00"])
    print(docs)