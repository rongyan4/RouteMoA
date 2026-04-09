import argparse
import importlib
import json
import logging
import os
import sys
import copy
from typing import List, Union, Optional, Dict
import uvicorn
from functools import partial
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from json import JSONDecodeError
from pydantic import BaseModel
from loguru import logger
# from lagent.schema import AgentMessage

from emoa.__version__ import version, title, description
from emoa.serve.routes import health, completion

# ---------------switch EmoaApp class---------------

#from emoa.core.singlemodel import EmoaApp
#from emoa.core.emoa_v2 import EmoaApp
from emoa.core.newemoa_v2 import EmoaApp

# ----------------

from transformers import AutoTokenizer
from emoa.router.src.models.mf import MatrixFactorizationRouter
from emoa.router.src.models.oracle import OracleRouter
from emoa.router.src.models.routerdc import RouterModule, RouterModuleSingle
import torch


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


class AsyncAgentAPIServer:
    def __init__(
            self,
            config,
            host: str = '0.0.0.0',
            port: int = 10010
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.app = FastAPI(
            docs_url='/'
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

        self.agent_config = config

        if config["mode"] == "moa":
            self.router = None
            self.router_tokenizer = None
        elif config["mode"] == "emoa":
            if config["router_type"] == "normal":
                router_backbone = config.pop("router_backbone")
                router_pth_path = config.pop("router_pth_path")
                num_labels = len(config["candidate_models"])
                self.router_tokenizer = AutoTokenizer.from_pretrained(router_backbone)
                model = MatrixFactorizationRouter(router_backbone, num_labels)  # todo: allow configuring different models
                model.load_state_dict(torch.load(router_pth_path))
                model.to(self.device)
                self.router = model
            elif config["router_type"] == "oracle":
                router_pth_path = config.pop("router_pth_path")
                self.router = OracleRouter(file_path=router_pth_path)
                self.router_tokenizer = None

            elif config["router_type"] == "routerdc":
                router_backbone = config.pop("router_backbone")
                router_pth_path = config.pop("router_pth_path")
                similarity_fn = config.pop("similarity_function", "cos")

                num_labels = len(config["candidate_models"])

                self.router_tokenizer = AutoTokenizer.from_pretrained(
                    router_backbone, truncation_side="left", padding=True, use_fast=False
                )
                from transformers import DebertaV2Model
                encoder_model = DebertaV2Model.from_pretrained(router_backbone)

                self.router = RouterModule(
                    backbone=encoder_model,
                    hidden_state_dim=encoder_model.config.hidden_size,
                    node_size=10,      
                    similarity_function=similarity_fn,
                )
                state_dict = torch.load(router_pth_path, map_location="cpu", weights_only=True)
                self.router.load_state_dict(state_dict, strict=True)
                self.router.to(self.device)

        self.setup_routes()
        self.run(host, port)

    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def process_message(request: ChatCompletionRequest):
            agent_config = copy.deepcopy(self.agent_config)
            req_dumps = request.model_dump(exclude_unset=False)
            logger.info(f"{req_dumps=}")

            agent_config.update(req_dumps)

            if "extra_body" in req_dumps:
                agent_config.update(req_dumps["extra_body"])
                del agent_config["extra_body"]
            # breakpoint()
            if len(agent_config["model"]) == 0:
                if "model" in self.agent_config:
                    agent_config["model"] = self.agent_config["model"]
                else:
                    raise ValueError("model not specified!")

            if "output_path" in agent_config:
                self.output_path = agent_config["output_path"]

            logger.info(f"{agent_config=}")

            agent = EmoaApp(
                router=self.router,
                router_tokenizer=self.router_tokenizer,
                **agent_config
            )

            try:
                # Ensure the agent call is correctly awaited and results are returned properly.
                result = await agent.run()
                print('result generated!!!!!!!!!!!!!!!!!!!!')
                # breakpoint()

                if isinstance(result, List):
                    if len(result) == 1:
                        response_obj = {
                            **result[0],
                            "request_body": agent_config,
                        }
                    else:
                        response_obj = {
                            "result": result,
                            "request_body": agent_config,
                        }
                elif isinstance(result, Dict):
                    response_obj = {
                        **result,
                        "request_body": agent_config,
                    }

                # await self.write_to_json(response_obj)

                # if self.output_path is not None:
                #     logger.info(f"Saving outputs to {self.output_path}.")

                #     with open(self.output_path, "a") as f:
                #         json.dump(response_obj, f, ensure_ascii=False)
                #         f.write("\n")

                json_compatible_item_data = jsonable_encoder(response_obj)

                return JSONResponse(content=json_compatible_item_data)
                # return response_obj

            except JSONDecodeError as e:
                logging.error(f'Error processing message: {str(e)}')
                #breakpoint()
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                #breakpoint()
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

        @self.app.get("/health")
        async def health_check():
            return {"message": "success"}

    def run(self, host='0.0.0.0', port=8090):
        logging.info(f'Starting server at {host}:{port}')
        uvicorn.run(self.app, host=host, port=port)

    async def write_to_json(self, response_obj):
        if self.output_path is not None:
            logger.info(f"Saving outputs to {self.output_path}.")

            with open(self.output_path, "a") as f:
                json.dump(response_obj, f, ensure_ascii=False)
                f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Async Agent API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=10010)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='JSON configuration for the agent')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    config = json.load(open(args.config, "r"))

    AsyncAgentAPIServer(config, host=args.host, port=args.port)
