"""
A thin proxy that exposes multiple logical model names while forwarding to a single
underlying OpenAI-compatible endpoint. It rewrites the requested model name to
`REAL_MODEL_NAME` and fakes `/v1/models` so upstream routers see the expected
candidates.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Logical model names the router expects to see.
LOGICAL_MODELS: List[str] = [
    "google/gemma-2-9b-it",
    "mistralai/Ministral-8B-Instruct-2410",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "ContactDoctor/Bio-Medical-Llama-3-8B",
]

# Downstream single model to actually call.
REAL_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DOWNSTREAM_BASE = "http://localhost:10403"
CHAT_COMPLETIONS_URL = f"{DOWNSTREAM_BASE}/v1/chat/completions"
MODELS_URL = f"{DOWNSTREAM_BASE}/v1/models"

app = FastAPI()
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


def _forward_headers(request: Request) -> Dict[str, str]:
    # Preserve Authorization so any downstream auth still works.
    auth = request.headers.get("authorization")
    return {"authorization": auth} if auth else {}


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    # Fake list showing logical names so the router believes these exist.
    data = [{"id": m, "object": "model"} for m in LOGICAL_MODELS]
    return JSONResponse({"object": "list", "data": data})


async def _stream_response(resp: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in resp.aiter_raw():
        if chunk:
            yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    assert _client is not None  # Satisfy type checker; managed in startup.
    payload = await request.json()
    payload["model"] = REAL_MODEL_NAME
    stream = bool(payload.get("stream", False))

    headers = _forward_headers(request)

    if stream:
        # Stream through chunks from downstream.
        resp = await _client.stream("POST", CHAT_COMPLETIONS_URL, json=payload, headers=headers)
        return StreamingResponse(_stream_response(resp), status_code=resp.status_code, headers={"content-type": resp.headers.get("content-type", "application/json")})

    resp = await _client.post(CHAT_COMPLETIONS_URL, json=payload, headers=headers)
    # Pass through status and body.
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# Convenience entrypoint: `python -m emoa.serve.model_alias_proxy`.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "emoa.serve.model_alias_proxy:app",
        host="0.0.0.0",
        port=18000,
        log_level="info",
    )
