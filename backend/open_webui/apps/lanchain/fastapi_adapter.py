from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Optional, Union,Any,Dict
from langchain_app import LangchainApp
from retriever import KnowledgeManager
import logging
log = logging.getLogger(__name__)
log.setLevel("debug")
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

knowledgeBase = KnowledgeManager(data_path="./vector_store")

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
        
        app = LangchainApp(model=model,retrievers=retriever)

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
        if r is not None:
            try:
                res = await r.json()
                if "error" in res:
                    error_detail = f"Ollama: {res['error']}"
            except Exception:
                error_detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status if r else 500,
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