import argparse
import json
import logging
import copy
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from emoa.core.smoa_v2 import SmoaApp 


# ============ Chat Data Structure ===========
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    extra_body: Optional[Dict] = {}


# ============ Fast API Server Class ==========
class AsyncAgentAPIServer:
    """FastAPI-based async server; nearly the same as the class in emoa.serve.app.py"""

    def __init__(self, config: Dict, host: str = "0.0.0.0", port: int = 10010):
        self.agent_config = config
        self.app = FastAPI(docs_url="/")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()
        self.run(host, port)

    # -------------------------- route --------------------------
    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def process_message(request: ChatCompletionRequest):
            cfg = copy.deepcopy(self.agent_config)
            req = request.model_dump(exclude_unset=False)
            logger.info(f"{req=}")

            cfg.update(req)
            if "extra_body" in req:          
                cfg.update(req["extra_body"])
                cfg.pop("extra_body", None)

            if not cfg.get("model"):
                raise ValueError("model not specified!")

            # try:
            #     agent = SmoaApp(**cfg)
            #     result = await agent.run()   

                
            #     if isinstance(result, list) and len(result) == 1:
            #         result = {**result[0], "request_body": cfg}
            #     elif isinstance(result, dict):
            #         result = {**result, "request_body": cfg}

            #     return JSONResponse(jsonable_encoder(result))

            # except Exception as e:
            #     logger.error(f"Error in request: {e}")
            #     raise HTTPException(status_code=500, detail="Internal Server Error")

            try:
                agent = SmoaApp(**cfg)
                result = await agent.run()
                
                if isinstance(result, list) and len(result) == 1:
                    result = {**result[0], "request_body": cfg}
                elif isinstance(result, dict):
                    result = {**result, "request_body": cfg}

                return JSONResponse(jsonable_encoder(result))

            except Exception as e:
                import traceback
                logger.error("Full exception traceback:\n" + traceback.format_exc())
                logger.error(f"Error in request: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")


        @self.app.get("/health")
        async def health_check():
            return {"message": "success"}

    # -------------------------- Run --------------------------
    def run(self, host: str, port: int):
        logging.info(f"Server running at {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# ============================== CLI args ==============================
def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async Agent API Server with SMoA")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=10010)
    p.add_argument("--config", required=True, help="JSON config file")

    # ---- prompt file ----
    p.add_argument("--role_prompt_file")
    p.add_argument("--judge_prompt_file")
    p.add_argument("--aggregation_prompt_file")

    # ---- SMoA paras chose ----
    p.add_argument("--add_role", action="store_true")
    p.add_argument("--moderate_select", action="store_true")
    p.add_argument("--moderate_end", action="store_true")
    p.add_argument("--num_select_response", type=int)

    return p.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = cli_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # CLI override → cfg
    for k in [
        "role_prompt_file",
        "judge_prompt_file",
        "aggregation_prompt_file",
        "add_role",
        "moderate_select",
        "moderate_end",
        "num_select_response",
    ]:
        v = getattr(args, k)
        if v not in (None, False):
            cfg[k] = v

    AsyncAgentAPIServer(cfg, host=args.host, port=args.port)