from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Optional, Union,Any,Dict
from langchain_app import LangchainApp

app = LangchainApp()

async def post_streaming_url(
    user_id: str,session_id: str, payload: Dict[str, Any], stream: bool = True, content_type=None
):
    input = None
    model = payload["model"]
    try:
        if payload["messages"]:
            input = payload["messages"][-1]["content"]
            
        if stream:
            headers = dict(r.headers)
            if content_type:
                headers["Content-Type"] = content_type
            return StreamingResponse(
                app.chat(input,user_id=user_id,conversation_id=session_id,stream=stream),
                status_code=r.status,
                headers=headers,
                background=BackgroundTask(
                    cleanup_response
                ),
            )
        else:
            res = app.chat(input,user_id=user_id,conversation_id=session_id,stream=stream)
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

