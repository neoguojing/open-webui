from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Optional, Union,Any,Dict
from langchain_app import LangchainApp

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



async def langchain_fastapi_wrapper(
    user_id: str,session_id: str, payload: Dict[str, Any], stream: bool = True, content_type="application/x-ndjson"
):
    input = None
    model = payload["model"]
    app = LangchainApp(model=model)

    try:
        if payload["messages"]:
            input = payload["messages"][-1]["content"]
        
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

