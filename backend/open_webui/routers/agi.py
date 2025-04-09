from openai import OpenAI
from contextlib import contextmanager
from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter,UploadFile
from fastapi.responses import FileResponse, StreamingResponse,JSONResponse
from open_webui.utils.auth import get_admin_user, get_verified_user
from typing import Optional
from open_webui.models.users import UserModel
from open_webui.env import ENV, SRC_LOG_LEVELS
from open_webui.config import AGI_API_KEY,AGI_BASE_URL
import requests
from aiocache import cached
import time
import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
router = APIRouter()

client = OpenAI(
    api_key=AGI_API_KEY, # This is the default and can be omitted
    base_url=AGI_BASE_URL,
)
@contextmanager
def handle_openai_errors():
    """捕获并处理 OpenAI 的常见异常"""
    try:
        yield  # 在实际调用中包裹 API 调用
    except Exception as e:
        # 捕获其他未知异常
        raise HTTPException(status_code=500, detail=str(e))

# 准备任务请求
async def prepare_task_params(form_data: dict):
    model = form_data.get("model")
    stream = form_data.get("stream")
    messages = form_data.get("messages")
    return {
        "model":model,
        "stream":stream,
        "messages":messages,
    }

# 准备业务请求
async def prepare_parmas(request,user):
    # 解析 JSON 请求体
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON request body")

    # 从请求体中提取各字段
    stream = data.get("stream", True)
    model = data.get("model", "agi")
    messages = data.get("messages", [])
    # params = data.get("params", {})
     # variables = data.get("variables", {})
    model_item = data.get("model_item", {})
    model_knowledge = model_item.get("info", {}).get("meta", {}).get("knowledge", False)
    # session_id = data.get("session_id")
    chat_id = data.get("chat_id")
    # request_id = data.get("id")
    files = data.get("files",[])
    
    # 处理知识库相关
    # 上传文件的知识库优先级比较高
    def merge_rag_info(files,model_knowledge):
        ret = []
        for f in files:
            if f.get("type") == "file":
                ret.append(f.get("collection_name",""))
        # 没有文件知识库的情况下才启用自带知识库
        if len(ret) == 0 and model_knowledge:
            for k in model_knowledge:
                if k.get("type") == "collection":
                    ret.append(f.get("id",""))

        return list(set(ret))
    # agi的知识库和openwebui隔离，相互不影响
    db_ids = merge_rag_info(files,model_knowledge)
    # 处理请求特性，适配agi
    features = data.get("features", {})
    feature = ""
    if features.get("web_search",False):
        feature = "web"
    elif db_ids and len(db_ids) > 0:
        feature = "rag"
    elif features.get("code_interpreter",False):
        pass
    elif features.get("image_generation",False):
        feature = "image2image"

    # 只处理最后一个消息，历史消息由agi自行处理，降低开销
    def convert_openai_message_to_agi_message(messages):
        message = messages[-1]
        if isinstance(message["content"],list):
            for item in message["content"]:
                if item.get("type") == "image_url":
                    item["type"] = "image"
                    item["image"] = item["image_url"]["url"]
                    item.pop("image_url",None)
        return [message]
    
    log.debug(f"agi resuqest: {messages}")
    return {
        "model":model,
        "stream":stream,
        "extra_body":{"db_ids":db_ids,"need_speech": False,"feature": feature,"conversation_id":chat_id},
        "user":user.id,
        "messages":convert_openai_message_to_agi_message(messages),
    }

@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    param = None
    if form_data.get("metadata") and form_data.get("metadata").get("task",None):
        param = await prepare_task_params(form_data)
    else:
        param = await prepare_parmas(request,user)
        
    stream = param.get("stream")
    
    def generate(param:dict):
        try:
            with handle_openai_errors():  # 处理异常
                model=param.get("model","agi")
                stream=param.get("stream",True)
                extra_body=param.get("extra_body",None)
                user=param.get("user","")
                messages=param.get("messages",[])
                # 调用 OpenAI 的流式 API（stream=True）
                response = client.chat.completions.create(
                    model=model,
                    stream=stream,
                    extra_body=extra_body,
                    user=user,
                    messages=messages,
                )
                
                if stream:
                    for event in response:
                        yield f"data: {event.to_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # 非流模式：直接返回完整响应（转换为 JSON 字符串）
                    log.debug(response)
                    
                    yield response.model_dump()

        except Exception as e:
            # 如果在流式过程中发生错误，返回错误信息
            yield f"data: {{\"error\": \"{e}\"}}\n\n"
    
    if stream:
        return StreamingResponse(
            generate(param),
            media_type="text/event-stream",  # Server-Sent Events (SSE)
        )
    else:
        # 对于非流模式
        # 取生成器的第一个输出
        result = next(generate(param))
        return result

# 处理agi返回的消息，以适配openwebui
# 返回text，则直接填充content
# 返回图片和语音等，则填充files
# content 改为markdown的模式，期望能更加方便的支持图片和音频展示
async def handle_agi_response(ret_content,event_emitter):
    files = []
    content = ""
    citations = []
    if isinstance(ret_content,str):
        content = ret_content
    elif isinstance(ret_content,list):
        ret_content = ret_content[0]
    
    if isinstance(ret_content,dict):
        if ret_content.get("type") == "text":
            content = ret_content.get("text","")
        elif ret_content.get("type") == "image":
            image_content = ret_content.get("image","")
            files.append({"type": "image","url": image_content})
            content = f"![Generated Image]({image_content})\n"
        elif ret_content.get("type") == "audio":
            audio_content = ret_content.get("audio","")
            audio_text = ret_content.get("text","")
            files.append({"type": "audio","url": audio_content})
            content = f'<audio controls><source src="{audio_content}" type="audio/mpeg">{audio_text}</audio>'
            
        citations = ret_content.get("citations",[])
    # 发送sources的事件
    if citations and len(citations) > 0 and event_emitter:
        log.info("got an citations {citations}")
        await event_emitter(
            {
                "type": "chat:completion",
                "data": {"sources": citations},
            }
        )

    return content


@cached(ttl=3)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info("get_all_models()")

    response = client.models.list()

    def extract_data(response):
        if response and "data" in response:
            return response["data"]
        if isinstance(response, list):
            return response
        return None
    models = []
    
    for model in response.data:
        models.append({
            "id": model.id,
            "name": model.id,
            "object": model.object,
            "created": int(time.time()),
            "owned_by": model.owned_by,
            "agi": model,
        })
        
    log.debug(f"models: {models}")
    return models 

# file文件需要单独存储到一个collection，以确保检索结果的准确
def upload_files(file_path: str,file_id: str,user_id: str, collection: str = None):
    from pathlib import Path
    url = f"{AGI_BASE_URL}/files"

    if collection is None:
        collection = f"file-{file_id}"
    files = {
        'file': (Path(file_path).name, open(file_path, 'rb'), 'application/octet-stream')
    }
    data = {
        "collection_name": collection,
        "user_id": user_id
    }

    try:
        response = requests.post(url, files=files, data=data, timeout=10)
        response.raise_for_status()  # 如果状态码不是 2xx，将引发异常
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "Failed to connect to the server"}
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}