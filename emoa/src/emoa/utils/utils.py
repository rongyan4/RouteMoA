import os
import json
import time
import requests
import openai
import copy
from openai import OpenAI

from loguru import logger

from emoa.utils import inject_references_to_message


DEBUG = int(os.environ.get("DEBUG", "0"))
