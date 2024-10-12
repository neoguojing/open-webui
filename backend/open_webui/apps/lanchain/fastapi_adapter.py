import os
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Optional, Union,Any,Dict
from llm_app import LangchainApp
from retriever import KnowledgeManager
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# {
# 	"model": "qwen2.5:14b",
# 	"created_at": "2024-10-08T13:54:01.747642283Z",
# 	"message": {
# 		"role": "assistant",
# 		"content": ""
# 	},
# 	"done_reason": "stop",
# 	"done": true,
# 	"total_duration": 1246690171,
# 	"load_duration": 11154470,
# 	"prompt_eval_count": 244,
# 	"prompt_eval_duration": 47987000,
# 	"eval_count": 28,
# 	"eval_duration": 491750000
# }

# {
# 	"model": "qwen2.5:14b",
# 	"created_at": "2024-10-08T13:54:01.255843212Z",
# 	"message": {
# 		"role": "assistant",
# 		"content": "Hello"
# 	},
# 	"done": false
# }

OLLAMA_BASE_URLS = os.environ.get("OLLAMA_BASE_URLS", "")
BASE_DIR = os.environ.get("BASE_DIR","./")
BACKEND_DIR = os.environ.get("BACKEND_DIR","./")
DATA_DIR = os.getenv("DATA_DIR", f"{BACKEND_DIR}/data")
CHROMA_DATA_PATH = f"{DATA_DIR}/vector_db"
LANGCHAIN_DB_PATH = f"sqlite:///{DATA_DIR}/langchain.db"

print("CHROMA_DATA_PATH:",CHROMA_DATA_PATH)
print("LANGCHAIN_DB_PATH:",LANGCHAIN_DB_PATH)
knowledgeBase = KnowledgeManager(data_path=CHROMA_DATA_PATH)

async def langchain_fastapi_wrapper(
    user_id: str,session_id: str, payload: Dict[str, Any], stream: bool = True, content_type="application/x-ndjson",topk=1
):
    input = None
    collections = None
    contexts = None
    model = payload["model"]
    app = None
    retriever = None

    try:
        files = payload.get("metadata", {}).get("files", None)
        if files:
            collections,contexts = get_rag_context(files)
            retriever = knowledgeBase.get_retriever(collections,topk)
        
        app = LangchainApp(model=model,retrievers=retriever,db_path=LANGCHAIN_DB_PATH)

        if payload["messages"]:
            input = get_last_user_message(payload["messages"])
        
        if stream:
            headers = {}
            if content_type:
                headers["Content-Type"] = content_type
            return StreamingResponse(
                app.ollama(input,user_id=user_id,conversation_id=session_id,stream=stream),
                status_code=200,
                headers=headers,
                background=BackgroundTask(
                    cleanup_response
                ),
            )
        else:
            res = app.ollama(input,user_id=user_id,conversation_id=session_id,stream=stream)
            await cleanup_response()
            return res

    except Exception as e:
        error_detail = "Open WebUI: Server Connection Error"
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )

async def cleanup_response():
    pass


def get_last_user_message_item(messages: list[dict]) -> Optional[dict]:
    for message in reversed(messages):
        if message["role"] == "user":
            return message
    return None


def get_content_from_message(message: dict) -> Optional[str]:
    if isinstance(message["content"], list):
        for item in message["content"]:
            if item["type"] == "text":
                return item["text"]
    else:
        return message["content"]
    return None


def get_last_user_message(messages: list[dict]) -> Optional[str]:
    message = get_last_user_message_item(messages)
    if message is None:
        return None
    return get_content_from_message(message)


def get_rag_context(
    files
):
    log.debug(f"files: {files}")

    extracted_collections = []
    relevant_contexts = []

    for file in files:
        context = None

        collection_names = (
            file["collection_names"]
            if file["type"] == "collection"
            else [file["collection_name"]] if file["collection_name"] else []
        )

        collection_names = set(collection_names).difference(extracted_collections)
        if not collection_names:
            log.debug(f"skipping {file} as it has already been extracted")
            continue

        if file["type"] == "text":
            context = file["content"]
            
        if context:
            relevant_contexts.append({**context, "source": file})

        extracted_collections.extend(collection_names)

    # contexts = []
    # citations = []

    # for context in relevant_contexts:
    #     try:
    #         if "documents" in context:
    #             contexts.append(
    #                 "\n\n".join(
    #                     [text for text in context["documents"][0] if text is not None]
    #                 )
    #             )

    #             if "metadatas" in context:
    #                 citations.append(
    #                     {
    #                         "source": context["source"],
    #                         "document": context["documents"][0],
    #                         "metadata": context["metadatas"][0],
    #                     }
    #                 )
    #     except Exception as e:
    #         log.exception(e)

    return extracted_collections,relevant_contexts

if __name__ == "__main__":
    nb = KnowledgeManager(data_path="/win/open-webui/backend/data/vector_db")
    # nb.store(collection_name="aaaaa",source="/home/neo/Downloads/ir2023_ashare.docx",file_name="ir2023_ashare.docx")
    retrievers = nb.get_retriever(collection_names="aaaaa",k=1)
    # retrievers = None
    app = LangchainApp(retrievers=retrievers,db_path="sqlite:////win/open-webui/backend/data/langchain.db")
    
    # resp = app("董事长报告书讲了什么？")
    # print("invoke:",resp)
    stream_generator = app.ollama("可持续发展")
    # 遍历生成器
    for response in stream_generator:
        print("iter:",response)
    # docs = nb.query_doc("aaaaa","董事长报告书")
    # print(docs)