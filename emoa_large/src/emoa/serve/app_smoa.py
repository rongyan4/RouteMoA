import argparse
import json
import logging
import copy
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger
import traceback
from pydantic import BaseModel

from emoa.core.smoa_v2 import SmoaApp


class ChatMessage(BaseModel):
    role: str
    content: str


class AsyncAgentAPIServer:
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

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def process_message(request: Request):
            req_dumps = await request.json()

            agent_config = copy.deepcopy(self.agent_config)
            logger.info(f"{req_dumps=}")
            agent_config.update(req_dumps)
            if "extra_body" in req_dumps:
                agent_config.update(req_dumps["extra_body"])
                agent_config.pop("extra_body", None)

            if not agent_config.get("model"):
                raise ValueError("model not specified!")

            try:
                agent = SmoaApp(**agent_config)
                result = await agent.run()

                if isinstance(result, list) and len(result) == 1:
                    result = {**result[0], "request_body": agent_config}
                elif isinstance(result, dict):
                    result = {**result, "request_body": agent_config}

                return JSONResponse(jsonable_encoder(result))

            except Exception as e:
                logger.error("Full exception traceback:\n" + traceback.format_exc())
                logger.error(f"Error in request: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @self.app.get("/health")
        async def health_check():
            return {"message": "success"}

        @self.app.get("/v1/models")
        async def show_available_models(request: Request):
            return JSONResponse({'object': 'list', 'data': [{'id': 'smoa'}]})

    def run(self, host: str, port: int):
        logging.info(f"Server running at {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMoA API Server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=10010)
    p.add_argument("--config", required=True, help="Path to JSON config file")

    # Prompt file overrides
    p.add_argument("--role_prompt_file")
    p.add_argument("--judge_prompt_file")
    p.add_argument("--aggregation_prompt_file")

    # SMoA parameter overrides
    p.add_argument("--add_role", action="store_true")
    p.add_argument("--moderate_select", action="store_true")
    p.add_argument("--moderate_end", action="store_true")
    p.add_argument("--num_select_response", type=int)

    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = cli_args()

    with open(args.config, "r") as f:
        agent_config = json.load(f)

    # CLI overrides → agent_config
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
            agent_config[k] = v

    AsyncAgentAPIServer(agent_config, host=args.host, port=args.port)
