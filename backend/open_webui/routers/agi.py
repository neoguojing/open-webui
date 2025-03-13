from openai import OpenAI
from contextlib import contextmanager
from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import FileResponse, StreamingResponse,JSONResponse
from open_webui.utils.auth import get_admin_user, get_verified_user
from typing import Optional
from open_webui.models.users import UserModel
from open_webui.env import ENV, SRC_LOG_LEVELS
from aiocache import cached
import time
import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
router = APIRouter()

client = OpenAI(
    api_key="123", # This is the default and can be omitted
    base_url="http://localhost:8000/v1",
)
@contextmanager
def handle_openai_errors():
    """捕获并处理 OpenAI 的常见异常"""
    try:
        yield  # 在实际调用中包裹 API 调用
    except client.error.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except client.error.RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail=f"请求过多，请稍后再试。{str(e)}",
        )
    except client.error.APIConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except client.error.APIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API 内部错误：{str(e)}",
        )
    except Exception as e:
        # 捕获其他未知异常
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
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
    def merge_rag_info(files,model_knowledge):
        ret = []
        for f in files:
            if f.get("type") == "file":
                ret.append(f.get("collection_name"))
        
        if model_knowledge:
            for k in model_knowledge:
                if k.get("type") == "collection":
                    ret.append(f.get("id"))

        return list(set(ret))
    db_ids = merge_rag_info(files,model_knowledge)

    # 处理请求特性，适配agi
    features = data.get("features", {})
    feature = "agent"
    if db_ids and len(db_ids) > 0:
        feature = "rag"
    elif features.get("web_search",False):
        feature = "web"
    elif features.get("code_interpreter",False):
        pass
    elif features.get("image_generation",False):
        pass

   

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
    
    def generate():
        try:
            with handle_openai_errors():  # 处理异常
                # 调用 OpenAI 的流式 API（stream=True）
                response = client.chat.completions.create(
                    model=model,
                    stream=stream,
                    extra_body={"db_ids":db_ids,"need_speech": False,"feature": feature,"conversation_id":chat_id},
                    user=user.id,
                    messages=convert_openai_message_to_agi_message(messages),
                )

                if stream:
                    for event in response:
                        yield f"data: {event.to_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # 非流模式：直接返回完整响应（转换为 JSON 字符串）
                    print("------------",type(response))
                    
                    yield response.to_json()

                

        except HTTPException as e:
            # 如果在流式过程中发生错误，返回错误信息
            yield f"data: {{\"error\": \"{e.detail}\"}}\n\n"
    
    if stream:
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",  # Server-Sent Events (SSE)
        )
    else:
        # 对于非流模式
        # 取生成器的第一个输出
        result = next(generate())
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
            citations = ret_content.get("citations",[])
        elif ret_content.get("type") == "image":
            image_content = ret_content.get("image","")
            files.append({"type": "image","url": image_content})
            content = f"![Generated Image]({image_content})\n"
        elif ret_content.get("type") == "audio":
            audio_content = ret_content.get("audio","")
            audio_text = ret_content.get("text","")
            files.append({"type": "audio","url": audio_content})
            content = f'<audio controls><source src="{audio_content}" type="audio/mpeg">{audio_text}</audio>'
    
    # 发送sources的事件
    if citations and len(citations) > 0 and event_emitter:
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
