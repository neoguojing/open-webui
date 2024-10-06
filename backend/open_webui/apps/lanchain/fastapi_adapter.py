from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from typing import Optional, Union
from langchain_app import LangchainApp

app = LangchainApp()

async def post_streaming_url(
    url: str, payload: Union[str, bytes], stream: bool = True, content_type=None
):
    r = None
    try:
        # session = aiohttp.ClientSession(
        #     trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        # )
        # r = await session.post(
        #     url,
        #     data=payload,
        #     headers={"Content-Type": "application/json"},
        # )
        # r.raise_for_status()

        if stream:
            headers = dict(r.headers)
            if content_type:
                headers["Content-Type"] = content_type
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=headers,
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            res = await r.json()
            await cleanup_response(r, session)
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

async def cleanup_response(
    response,
    session,
):
    if response:
        response.close()
    if session:
        await session.close()

