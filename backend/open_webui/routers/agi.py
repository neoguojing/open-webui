from openai import OpenAI
from contextlib import contextmanager
from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import FileResponse, StreamingResponse,JSONResponse
from open_webui.utils.auth import get_admin_user, get_verified_user
from typing import Optional
import json
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
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # 非流模式：直接返回完整响应（转换为 JSON 字符串）
                    # yield json.dumps(response, ensure_ascii=False)
                    yield response

                

        except HTTPException as e:
            # 如果在流式过程中发生错误，返回错误信息
            yield f"data: {{\"error\": \"{e.detail}\"}}\n\n"
    
    if stream:
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",  # Server-Sent Events (SSE)
        )
    else:
        # 对于非流模式，直接返回 JSONResponse
        # 取生成器的第一个输出
        result = next(generate())
        # 将 JSON 字符串转换为 Python 对象返回
        # return JSONResponse(content=json.loads(result))
        return result

