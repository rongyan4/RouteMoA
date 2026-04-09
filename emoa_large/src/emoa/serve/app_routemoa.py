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
from json import JSONDecodeError
from pydantic import BaseModel
from loguru import logger
import traceback

from emoa.__version__ import version, title, description
from emoa.core.emoa_v2 import EmoaApp

from transformers import AutoTokenizer, DebertaV2Model
from emoa.router.router import RouterModule
import torch


class ChatMessage(BaseModel):
    role: str
    content: str


class AsyncAgentAPIServer:
    def __init__(
            self,
            config,
            host: str = '0.0.0.0',
            port: int = 10010
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.app = FastAPI(docs_url='/')
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

        self.agent_config = config

        router_backbone = config.pop("router_backbone")
        router_pth_path = config.pop("router_pth_path")
        similarity_fn = config.pop("similarity_function", "cos")

        self.router_tokenizer = AutoTokenizer.from_pretrained(
            router_backbone, truncation_side="left", padding=True, use_fast=False
        )
        encoder_model = DebertaV2Model.from_pretrained(router_backbone)

        self.router = RouterModule(
            backbone=encoder_model,
            hidden_state_dim=encoder_model.config.hidden_size,
            node_size=16,
            similarity_function=similarity_fn,
        )
        state_dict = torch.load(router_pth_path, map_location="cpu", weights_only=True)
        self.router.load_state_dict(state_dict, strict=True)
        self.router.to(self.device)

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
                del agent_config["extra_body"]
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
                result = await agent.run()
                print("**************** Generation Succeeded ****************")

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

                json_compatible_item_data = jsonable_encoder(response_obj)
                return JSONResponse(content=json_compatible_item_data)

            except JSONDecodeError as e:
                logging.error(f'Error processing message: {str(e)}')
                logging.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail='Internal Server Error')

            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                logging.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail='Internal Server Error')

        @self.app.get("/health")
        async def health_check():
            return {"message": "success"}

        @self.app.get("/v1/models")
        async def show_available_models(request: Request):
            return JSONResponse({'object': 'list', 'data': [{'id': 'routemoa'}]})

    def run(self, host='0.0.0.0', port=8090):
        logging.info(f'Starting server at {host}:{port}')
        uvicorn.run(self.app, host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='RouteMoA API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=10010)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to JSON configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    AsyncAgentAPIServer(config, host=args.host, port=args.port)
