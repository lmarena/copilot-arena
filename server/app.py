import os
import time
import numpy as np
import uuid
import traceback
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi.responses import StreamingResponse, JSONResponse, Response
from apis.clients import (
    IBaseClient,
    OpenAIClient,
    MistralClient,
    DeepseekClient,
    DeepseekFimClient,
    AnthropicClient,
    GeminiClient,
    FireworksClient,
)
from src.utils import get_settings, get_models_by_tags, get_cost
from apis.base_client import State, LLMOptions
from src.firebase_client import FirebaseClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from constants import (
    MAX_LINES,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    MAX_INPUT_TOKENS,
    PREFIX_RATIO,
)
from src.privacy import PrivacySetting, privacy_aware_log
from json.decoder import JSONDecodeError
import tiktoken
from src.scores import get_scores
from amplitude import Amplitude, BaseEvent
from pydantic import BaseModel, Field
from typing import Optional
from functools import wraps
from src.user_repository import (
    UserRepository,
    UserNotFoundError,
    UsernameExistsError,
    PasswordIncorrectError,
)


try:
    from config.amplitude_config import AMPLITUDE_API_KEY
except ImportError:
    AMPLITUDE_API_KEY = os.environ.get("AMPLITUDE_API_KEY")

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)


### prometheus metrics ###
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"]
)
ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP Errors",
    ["method", "endpoint", "exception", "status"],
)
MODEL_CREATE_COUNT = Counter(
    "model_create_total", "Total Model Create Calls", ["client", "model", "status"]
)
MODEL_CREATE_LATENCY = Histogram(
    "model_create_complete_latency_seconds", "Model Create Latency", ["client", "model"]
)


async def timed_create(client, state, model, options):
    try:
        print(f"Starting {client.__class__.__name__}.create for model {model}")
        start_time = asyncio.get_event_loop().time()
        result = await client.create(state, model, options)
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        print(
            f"Finished {client.__class__.__name__}.create for model {model}. Took {latency:.4f} seconds"
        )
        print(f"result {model}: {result}")
        MODEL_CREATE_COUNT.labels(
            client=client.__class__.__name__, model=model, status="success"
        ).inc()
        MODEL_CREATE_LATENCY.labels(
            client=client.__class__.__name__, model=model
        ).observe(latency)
        return result, latency
    except Exception as e:
        print(f"Error caught in timed_create by {model}: {e}")
        MODEL_CREATE_COUNT.labels(
            client=client.__class__.__name__, model=model, status="error"
        ).inc()
        raise e


# Used exclusively for load testing
async def timed_create_test(client, state, model, options):
    print(f"MOCK: Starting {client.__class__.__name__}.create for model {model}")
    start_time = asyncio.get_event_loop().time()
    # Wait for 500ms to 1500 ms
    result = await asyncio.sleep(np.random.uniform(0.5, 1.5))
    result = "test"
    end_time = asyncio.get_event_loop().time()
    latency = end_time - start_time
    print(
        f"MOCK: Finished {client.__class__.__name__}.create for model {model}. Took {latency:.4f} seconds"
    )
    return result, latency


# Add these two new constants
PROMPT_INDEX_EDIT = 1
PROMPT_INDEX_FIM = 0


class UserMetadata(BaseModel):
    jobTitle: Optional[str] = None
    yearsOfExperience: Optional[int] = None
    codingHoursPerWeek: Optional[int] = None
    industrySector: Optional[str] = None


class User(BaseModel):
    user_id: str = Field(..., alias="userId")
    username: str = Field(..., alias="username")
    password: Optional[str] = Field(None, alias="password")
    metadata: Optional[UserMetadata] = None


class UpdateUser(BaseModel):
    user_id: Optional[str] = Field(..., alias="userId")
    username: str = Field(..., alias="username")
    old_password: Optional[str] = Field(None, alias="oldPassword")
    new_password: Optional[str] = Field(None, alias="newPassword")
    metadata: Optional[UserMetadata] = None


# Add this new function near the top of the file, after imports
def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = next((arg for arg in args if isinstance(arg, Request)), None)
        method = request.method if request else "UNKNOWN"
        endpoint = request.url.path if request else f"/{func.__name__}"
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            if "429" in str(e):
                logger.error("Rate limit exceeded.")
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    exception="RateLimitExceeded",
                    status=429,
                ).inc()
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later.",
                )
            elif "timed out" in str(e).lower() or "timeout" in str(e).lower():
                logger.error("API timeout occurred.")
                ERROR_COUNT.labels(
                    method=method, endpoint=endpoint, exception="Timeout", status=504
                ).inc()
                raise HTTPException(
                    status_code=504,
                    detail="API request timed out. Please try again later.",
                )
            else:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

    return wrapper


class FastAPIApp:
    def __init__(self):
        self.FIREBASE_COLLECTIONS_KEY = "firebase_collections"
        self.amplitude = Amplitude(AMPLITUDE_API_KEY)
        self.settings = get_settings()
        self.version_backend = self.settings.get("version_backend")
        self.limiter = Limiter(key_func=get_remote_address)
        self.app = FastAPI()
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.firebase_client = FirebaseClient()
        # Set up user repository (Must come after firebase client for auth)
        self.user_repo = UserRepository()

        # Set up all clients
        self.model_clients = {}
        self.tag_to_models = {}
        self.add_client(OpenAIClient())
        self.add_client(AnthropicClient())
        self.add_client(GeminiClient())
        self.add_client(FireworksClient())
        self.add_client(DeepseekClient())
        # FiM Clients
        self.add_client(MistralClient())
        self.add_client(DeepseekFimClient())
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        logger.info(f"models: {self.models}")

        # Extract weights for active models (not commented out)
        self.model_weights = {
            model_name: model_info["weight"]
            for model_name, model_info in self.settings["models"].items()
        }

        # Calculate total weight for normalization
        total_weight = sum(self.model_weights.values())

        # Calculate normalized probabilities
        self.model_probabilities = {
            model_name: weight / total_weight
            for model_name, weight in self.model_weights.items()
        }
        print(self.model_probabilities)

        # Initialize global_outcomes_df
        self.update_global_outcomes_df()

        # Set up scheduler for updating global_outcomes_df
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.update_global_outcomes_df,
            trigger=IntervalTrigger(hours=1),
            id="update_global_outcomes_df",
            name="Update global outcomes DataFrame every hour",
            replace_existing=True,
        )
        self.scheduler.start()

        self.setup_routes()
        logger.info("API is starting up")

    def update_global_outcomes_df(self):
        autocomplete_outcomes_collection_name = self.settings[
            self.FIREBASE_COLLECTIONS_KEY
        ]["outcomes"]
        self.global_outcomes_df = self.firebase_client.get_autocomplete_outcomes(
            autocomplete_outcomes_collection_name
        )
        logger.info("Updated global_outcomes_df")

    def add_client(self, client: IBaseClient):
        for model in client.models:
            if model in self.settings["models"]:
                self.model_clients[model] = client
                for tag in self.settings["models"][model]["tags"]:
                    if tag not in self.tag_to_models:
                        self.tag_to_models[tag] = set()
                    self.tag_to_models[tag].add(model)

        self.models = list(self.model_clients.keys())

    def select_models(self, tags):
        tagged_models = get_models_by_tags(tags, self.models, self.tag_to_models)
        if len(tagged_models) == 0:
            tagged_models = self.models
        model_probabilities = [
            self.model_probabilities[model] for model in tagged_models
        ]

        model_probabilities = np.array(model_probabilities) / sum(model_probabilities)

        selected_models = np.random.choice(
            tagged_models, size=2, replace=False, p=model_probabilities
        ).tolist()

        # Randomly shuffle the order of selected models
        if np.random.random() < 0.5:
            selected_models = selected_models[::-1]

        client1: IBaseClient = self.model_clients.get(selected_models[0])
        client2: IBaseClient = self.model_clients.get(selected_models[1])
        return selected_models, client1, client2

    def setup_routes(self):
        @self.app.post("/stream_complete")
        @self.limiter.limit("1000/minute")
        async def stream_complete(request: Request):
            # TODO prompt length truncation
            try:
                data = await request.json()
            except JSONDecodeError:
                error_msg = "Invalid JSON in request body"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/stream_complete",
                    exception="JSONDecodeError",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            if "prompt" not in data or "model" not in data:
                error_msg = "Request must contain 'prompt' and 'model'"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/stream_complete",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            prompt = data.get("prompt")
            model = data.get("model")
            temperature = data.get("temperature", 0.5)
            max_tokens = data.get("max_tokens", 1024)
            top_p = data.get("top_p", 1.0)

            client: IBaseClient = self.model_clients.get(model)

            async def stream_generator():
                async for part in client.stream(
                    prompt, model, temperature, max_tokens, top_p
                ):
                    yield part

            return StreamingResponse(stream_generator(), media_type="text/plain")

        @self.app.post("/create_edit_pair")
        @self.limiter.limit("300/minute")
        @handle_exceptions
        async def create_edit_pair(request: Request, background_tasks: BackgroundTasks):
            start_time = time.time()
            latency_breakdown = {}

            # Request parsing
            request_parse_start = time.time()
            try:
                data = await request.json()
            except JSONDecodeError:
                error_msg = "Invalid JSON in request body"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_edit_pair",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            if (
                "prefix" not in data
                or "codeToEdit" not in data
                or "userInput" not in data
                or "language" not in data
                or "userId" not in data
                or "suffix" not in data
                or "privacy" not in data
            ):
                error_msg = "Request must contain keys: [prefix, codeToEdit, userInput, userId, privacy, suffix]"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_edit_pair",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            try:
                privacy = PrivacySetting(data.get("privacy"))
            except ValueError:
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_edit_pair",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}",
                )

            latency_breakdown["request_parsing"] = time.time() - request_parse_start

            # Data preparation
            data_prep_start = time.time()
            pairId = data.get("pairId", str(uuid.uuid4()))
            prefix = data.get("prefix")
            code_to_edit = data.get("codeToEdit")
            user_input = data.get("userInput")
            language = data.get("language")
            suffix = data.get("suffix")
            user_id = data.get("userId")
            tags = data.get("modelTags", [])
            tags.append("edit")  # always add the edit tag

            background_tasks.add_task(
                self.amplitude.track,
                BaseEvent(event_type="create_pair", user_id=user_id),
            )

            state = State(
                prefix=prefix,
                suffix=suffix,
                code_to_edit=code_to_edit,
                language=language,
                user_input=user_input,
                full_prefix=prefix,
                full_suffix=suffix,
            )
            options = LLMOptions(
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                max_lines=100,
                prompt_index=PROMPT_INDEX_EDIT,  # Use the new constant here
            )
            latency_breakdown["data_preparation"] = time.time() - data_prep_start

            # Model selection
            model_selection_start = time.time()
            selected_models, client1, client2 = self.select_models(tags)
            latency_breakdown["model_selection"] = time.time() - model_selection_start

            # Handle prefix and suffix max lengths
            prefix_max_tokens = int(MAX_INPUT_TOKENS * PREFIX_RATIO)
            suffix_max_tokens = MAX_INPUT_TOKENS - prefix_max_tokens
            prefix_tokens = self.encoding.encode(state.prefix, allowed_special="all")
            code_to_edit_tokens = self.encoding.encode(
                state.prefix, allowed_special="all"
            )
            suffix_tokens = self.encoding.encode(state.suffix, allowed_special="all")
            full_prefix_length = len(prefix_tokens)
            full_suffix_length = len(suffix_tokens)
            prefix_tokens = prefix_tokens[-prefix_max_tokens:]
            suffix_tokens = suffix_tokens[:suffix_max_tokens]
            state.prefix = self.encoding.decode(prefix_tokens)
            state.suffix = self.encoding.decode(suffix_tokens)

            # Concurrent task execution
            task_execution_start = time.time()
            task1 = asyncio.create_task(
                timed_create(
                    client=client1,
                    state=state,
                    model=selected_models[0],
                    options=options,
                )
            )
            task2 = asyncio.create_task(
                timed_create(
                    client=client2,
                    state=state,
                    model=selected_models[1],
                    options=options,
                )
            )
            (response1, latency1), (response2, latency2) = await asyncio.gather(
                task1, task2
            )
            latency_breakdown["task_execution"] = time.time() - task_execution_start
            latency_breakdown["model1_latency"] = latency1
            latency_breakdown["model2_latency"] = latency2

            # Response preparation
            response_prep_start = time.time()
            responseId1, responseId2 = str(uuid.uuid4()), str(uuid.uuid4())
            timestamp = int(datetime.now().timestamp())
            prompt1 = client1.generate_prompt_for_model(
                state=state,
                model=selected_models[0],
                prompt_index=PROMPT_INDEX_EDIT,  # Use the new constant here
            )
            prompt2 = client2.generate_prompt_for_model(
                state=state,
                model=selected_models[1],
                prompt_index=PROMPT_INDEX_EDIT,  # Use the new constant here
            )
            latency_breakdown["response_preparation"] = (
                time.time() - response_prep_start
            )

            encoding_start = time.time()
            # Generate encodings for prompt1,2,prefix,suffix
            # check if string
            try:
                if isinstance(prompt1, str):
                    prompt1_tokens = self.encoding.encode(
                        prompt1, allowed_special="all"
                    )
                    prompt1_length = len(prompt1)
                else:
                    prompt1_tokens = []
                    prompt1_length = 0
                    for msg in prompt1:
                        content = msg["content"]
                        prompt1_tokens.extend(
                            self.encoding.encode(content, allowed_special="all")
                        )
                        prompt1_length += len(content)

                if isinstance(prompt2, str):
                    prompt2_tokens = self.encoding.encode(
                        prompt2, allowed_special="all"
                    )
                    prompt2_length = len(prompt2)
                else:
                    prompt2_tokens = []
                    prompt2_length = 0
                    for msg in prompt2:
                        content = msg["content"]
                        prompt2_tokens.extend(
                            self.encoding.encode(content, allowed_special="all")
                        )
                        prompt2_length += len(content)

            except Exception as e:
                logger.error(e)
                prompt1_length = 0
                prompt2_length = 0
                prompt1_tokens = []
                prompt2_tokens = []
                prefix_tokens = []
                suffix_tokens = []

            response1_tokens = self.encoding.encode(response1, allowed_special="all")
            response2_tokens = self.encoding.encode(response2, allowed_special="all")

            latency_breakdown["encoding"] = time.time() - encoding_start

            model_1_cost = get_cost(
                model=selected_models[0],
                prompt_token_length=len(prompt1_tokens),
                response_token_length=len(response1_tokens),
            )
            model_2_cost = get_cost(
                model=selected_models[1],
                prompt_token_length=len(prompt2_tokens),
                response_token_length=len(response2_tokens),
            )

            responseItem1 = {
                "responseId": responseId1,
                "userId": user_id,
                "timestamp": timestamp,
                "prompt": prompt1,
                "prefix": state.prefix,
                "code_to_edit": state.code_to_edit,
                "user_input": state.user_input,
                "suffix": state.suffix,
                "full_prefix": state.full_prefix,
                "full_suffix": state.full_suffix,
                "prompt_token_length": len(prompt1_tokens),
                "prefix_token_length": len(prefix_tokens),
                "code_to_edit_token_length": len(code_to_edit_tokens),
                "suffix_token_length": len(suffix_tokens),
                "full_prefix_length": full_prefix_length,
                "full_suffix_length": full_suffix_length,
                "response_token_length": len(response1_tokens),
                "model_1_cost": model_1_cost,
                "latency": latency1,
                "response": response1,
                "model": selected_models[0],
                "pairResponseId": responseId2,
                "pairIndex": 0,
                "privacy": privacy,
                "versionBackend": self.version_backend,
            }
            responseItem2 = {
                "responseId": responseId2,
                "userId": user_id,
                "timestamp": timestamp,
                "prompt": prompt2,
                "prefix": state.prefix,
                "code_to_edit": state.code_to_edit,
                "user_input": state.user_input,
                "suffix": state.suffix,
                "full_prefix": state.full_prefix,
                "full_suffix": state.full_suffix,
                "prompt_token_length": len(prompt2_tokens),
                "prefix_token_length": len(prefix_tokens),
                "code_to_edit_token_length": len(code_to_edit_tokens),
                "suffix_token_length": len(suffix_tokens),
                "full_prefix_length": full_prefix_length,
                "full_suffix_length": full_suffix_length,
                "response_token_length": len(response2_tokens),
                "model_2_cost": model_2_cost,
                "latency": latency2,
                "response": response2,
                "model": selected_models[1],
                "pairResponseId": responseId1,
                "pairIndex": 1,
                "privacy": privacy,
                "versionBackend": self.version_backend,
            }

            json_content = {
                "pairId": pairId,
                "responseItems": [responseItem1, responseItem2],
            }
            privacy_aware_log(json_content, privacy, logger, logging.INFO)

            # Firebase upload
            firebase_upload_start = time.time()
            collection = self.settings[self.FIREBASE_COLLECTIONS_KEY]["edits"]
            background_tasks.add_task(
                self.amplitude.track,
                BaseEvent(
                    event_type="create_edit_pair_success",
                    user_id=user_id,
                    event_properties={
                        "model_1": selected_models[0],
                        "model_2": selected_models[1],
                        "model_1_tokens": len(prompt1_tokens),
                        "model_2_tokens": len(prompt2_tokens),
                        "model_1_cost": model_1_cost,
                        "model_2_cost": model_2_cost,
                        "cost": model_1_cost + model_2_cost,
                    },
                ),
            )
            background_tasks.add_task(
                self.firebase_client.upload_data,
                collection,
                responseItem1,
                privacy,
            )
            background_tasks.add_task(
                self.firebase_client.upload_data,
                collection,
                responseItem2,
                privacy,
            )
            latency_breakdown["firebase_upload"] = time.time() - firebase_upload_start

            # Final response
            response = JSONResponse(content=json_content)
            end_time = time.time()
            total_latency = end_time - start_time
            latency_breakdown["total_latency"] = total_latency

            logger.info(latency_breakdown)
            logger.info(
                f"Total Latency: {total_latency:.4f}s, model {selected_models[0]}: {latency1:.4f}s, model {selected_models[1]}: {latency2:.4f}s | {responseId1} | {responseId2}"
            )

            return response

        @self.app.post("/create_pair")
        @self.limiter.limit("300/minute")
        async def create_pair(request: Request, background_tasks: BackgroundTasks):
            start_time = time.time()
            latency_breakdown = {}

            # Request parsing
            request_parse_start = time.time()
            try:
                data = await request.json()
            except JSONDecodeError:
                error_msg = "Invalid JSON in request body"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_pair",
                    exception="JSONDecodeError",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            if "prefix" not in data or "userId" not in data or "privacy" not in data:
                error_msg = "Request must contain keys: [prefix, userId, privacy]"
                logger.info(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_pair",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(status_code=400, detail=error_msg)

            try:
                privacy = PrivacySetting(data.get("privacy"))
            except ValueError:
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_pair",
                    exception="BadRequest",
                    status=400,
                ).inc()
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}",
                )

            latency_breakdown["request_parsing"] = time.time() - request_parse_start

            # Data preparation
            data_prep_start = time.time()
            pairId = data.get("pairId", str(uuid.uuid4()))
            temperature = data.get("temperature", TEMPERATURE)
            max_tokens = data.get("max_tokens", data.get("maxTokens", MAX_TOKENS))
            top_p = data.get("top_p", data.get("topP", TOP_P))
            max_lines = data.get("max_lines", data.get("maxLines", MAX_LINES))
            prefix = data.get("prefix")
            suffix = data.get("suffix", "")
            user_id = data.get("userId")
            tags = data.get("modelTags", [])

            background_tasks.add_task(
                self.amplitude.track,
                BaseEvent(event_type="create_pair", user_id=user_id),
            )

            state = State(
                prefix=prefix,
                suffix=suffix,
                full_prefix=prefix,
                full_suffix=suffix,
            )
            options = LLMOptions(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                max_lines=max_lines,
                prompt_index=PROMPT_INDEX_FIM,
            )
            latency_breakdown["data_preparation"] = time.time() - data_prep_start

            # Model selection
            model_selection_start = time.time()
            selected_models, client1, client2 = self.select_models(tags)
            latency_breakdown["model_selection"] = time.time() - model_selection_start

            # Handle prefix and suffix max lengths
            prefix_max_tokens = int(MAX_INPUT_TOKENS * PREFIX_RATIO)
            suffix_max_tokens = MAX_INPUT_TOKENS - prefix_max_tokens
            prefix_tokens = self.encoding.encode(state.prefix, allowed_special="all")
            suffix_tokens = self.encoding.encode(state.suffix, allowed_special="all")
            full_prefix_length = len(prefix_tokens)
            full_suffix_length = len(suffix_tokens)
            prefix_tokens = prefix_tokens[-prefix_max_tokens:]
            suffix_tokens = suffix_tokens[:suffix_max_tokens]
            state.prefix = self.encoding.decode(prefix_tokens)
            state.suffix = self.encoding.decode(suffix_tokens)

            # Concurrent task execution
            task_execution_start = time.time()
            task1 = asyncio.create_task(
                timed_create(
                    client=client1,
                    state=state,
                    model=selected_models[0],
                    options=options,
                )
            )
            task2 = asyncio.create_task(
                timed_create(
                    client=client2,
                    state=state,
                    model=selected_models[1],
                    options=options,
                )
            )
            (completion1, latency1), (completion2, latency2) = await asyncio.gather(
                task1, task2
            )
            latency_breakdown["task_execution"] = time.time() - task_execution_start
            latency_breakdown["model1_latency"] = latency1
            latency_breakdown["model2_latency"] = latency2

            # Response preparation
            response_prep_start = time.time()
            completionId1, completionId2 = str(uuid.uuid4()), str(uuid.uuid4())
            timestamp = int(datetime.now().timestamp())
            prompt1 = client1.generate_prompt_for_model(
                state=state,
                model=selected_models[0],
                prompt_index=PROMPT_INDEX_FIM,  # Use the new constant here
            )
            prompt2 = client2.generate_prompt_for_model(
                state=state,
                model=selected_models[1],
                prompt_index=PROMPT_INDEX_FIM,  # Use the new constant here
            )
            latency_breakdown["response_preparation"] = (
                time.time() - response_prep_start
            )

            encoding_start = time.time()
            # Generate encodings for prompt1,2,prefix,suffix
            # check if string
            try:
                if isinstance(prompt1, str):
                    prompt1_tokens = self.encoding.encode(
                        prompt1, allowed_special="all"
                    )
                    prompt1_length = len(prompt1)
                else:
                    prompt1_tokens = []
                    prompt1_length = 0
                    for msg in prompt1:
                        content = msg["content"]
                        prompt1_tokens.extend(
                            self.encoding.encode(content, allowed_special="all")
                        )
                        prompt1_length += len(content)

                if isinstance(prompt2, str):
                    prompt2_tokens = self.encoding.encode(
                        prompt2, allowed_special="all"
                    )
                    prompt2_length = len(prompt2)
                else:
                    prompt2_tokens = []
                    prompt2_length = 0
                    for msg in prompt2:
                        content = msg["content"]
                        prompt2_tokens.extend(
                            self.encoding.encode(content, allowed_special="all")
                        )
                        prompt2_length += len(content)

            except Exception as e:
                logger.error(e)
                prompt1_length = 0
                prompt2_length = 0
                prompt1_tokens = []
                prompt2_tokens = []
                prefix_tokens = []
                suffix_tokens = []

            completion1_tokens = self.encoding.encode(
                completion1, allowed_special="all"
            )
            completion2_tokens = self.encoding.encode(
                completion2, allowed_special="all"
            )

            latency_breakdown["encoding"] = time.time() - encoding_start

            model_1_cost = get_cost(
                model=selected_models[0],
                prompt_token_length=len(prompt1_tokens),
                response_token_length=len(completion1_tokens),
            )
            model_2_cost = get_cost(
                model=selected_models[1],
                prompt_token_length=len(prompt2_tokens),
                response_token_length=len(completion2_tokens),
            )

            completionItem1 = {
                "completionId": completionId1,
                "userId": user_id,
                "timestamp": timestamp,
                "prompt": prompt1,
                "prefix": state.prefix,
                "suffix": state.suffix,
                "full_prefix": state.full_prefix,
                "full_suffix": state.full_suffix,
                "prompt_length": prompt1_length,
                "prefix_length": len(state.prefix),
                "suffix_length": len(state.suffix),
                "completion_length": len(completion1),
                "prompt_token_length": len(prompt1_tokens),
                "prefix_token_length": len(prefix_tokens),
                "suffix_token_length": len(suffix_tokens),
                "full_prefix_length": full_prefix_length,
                "full_suffix_length": full_suffix_length,
                "completion_token_length": len(completion1_tokens),
                "model_1_cost": model_1_cost,
                "latency": latency1,
                "completion": completion1,
                "model": selected_models[0],
                "pairCompletionId": completionId2,
                "pairIndex": 0,
                "privacy": privacy,
                "versionBackend": self.version_backend,
            }
            completionItem2 = {
                "completionId": completionId2,
                "userId": user_id,
                "timestamp": timestamp,
                "prompt": prompt2,
                "prefix": state.prefix,
                "suffix": state.suffix,
                "full_prefix": state.full_prefix,
                "full_suffix": state.full_suffix,
                "prompt_length": prompt2_length,
                "prefix_length": len(state.prefix),
                "suffix_length": len(state.suffix),
                "completion_length": len(completion2),
                "prompt_token_length": len(prompt2_tokens),
                "prefix_token_length": len(prefix_tokens),
                "suffix_token_length": len(suffix_tokens),
                "full_prefix_length": full_prefix_length,
                "full_suffix_length": full_suffix_length,
                "completion_token_length": len(completion2_tokens),
                "model_2_cost": model_2_cost,
                "latency": latency2,
                "completion": completion2,
                "model": selected_models[1],
                "pairCompletionId": completionId1,
                "pairIndex": 1,
                "privacy": privacy,
                "versionBackend": self.version_backend,
            }

            json_content = {
                "pairId": pairId,
                "completionItems": [completionItem1, completionItem2],
            }
            privacy_aware_log(json_content, privacy, logger, logging.INFO)

            # Firebase upload
            firebase_upload_start = time.time()
            collection = self.settings[self.FIREBASE_COLLECTIONS_KEY]["all_completions"]
            background_tasks.add_task(
                self.amplitude.track,
                BaseEvent(
                    event_type="create_pair_success",
                    user_id=user_id,
                    event_properties={
                        "model_1": selected_models[0],
                        "model_2": selected_models[1],
                        "model_1_tokens": len(prompt1_tokens),
                        "model_2_tokens": len(prompt2_tokens),
                        "model_1_cost": model_1_cost,
                        "model_2_cost": model_2_cost,
                        "cost": model_1_cost + model_2_cost,
                    },
                ),
            )
            background_tasks.add_task(
                self.firebase_client.upload_data,
                collection,
                completionItem1,
                privacy,
            )
            background_tasks.add_task(
                self.firebase_client.upload_data,
                collection,
                completionItem2,
                privacy,
            )
            latency_breakdown["firebase_upload"] = time.time() - firebase_upload_start

            # Final response
            response = JSONResponse(content=json_content)
            end_time = time.time()
            total_latency = end_time - start_time
            latency_breakdown["total_latency"] = total_latency

            logger.info(latency_breakdown)
            logger.info(
                f"Total Latency: {total_latency:.4f}s, model {selected_models[0]}: {latency1:.4f}s, model {selected_models[1]}: {latency2:.4f}s | {completionId1} | {completionId2}"
            )

            return response

        @self.app.put("/add_completion")
        @self.limiter.limit("1000/minute")
        async def add_completion(request: Request, background_tasks: BackgroundTasks):
            try:
                collection = self.settings[self.FIREBASE_COLLECTIONS_KEY]["completions"]
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                required_keys = [
                    "completionId",
                    "pairCompletionId",
                    "pairIndex",
                    "userId",
                    "timestamp",
                    "prompt",
                    "completion",
                    "model",
                    "version",
                    "privacy",
                ]

                # if any of the required keys are missing, return 400
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                try:
                    privacy = PrivacySetting(data.get("privacy"))
                except ValueError:
                    error_msg = f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                data["versionBackend"] = self.version_backend

                if data["model"] == "test":
                    return {"status": "success"}

                background_tasks.add_task(
                    self.amplitude.track,
                    BaseEvent(
                        event_type="add_completion",
                        user_id=data["userId"],
                        event_properties={"model": data["model"]},
                    ),
                )
                background_tasks.add_task(
                    self.firebase_client.upload_data, collection, data, privacy
                )
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/add_completion",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        @self.app.put("/add_single_outcome")
        @self.limiter.limit("1000/minute")
        async def add_single_outcome(request: Request):
            try:
                collection = self.settings[self.FIREBASE_COLLECTIONS_KEY][
                    "single_outcomes"
                ]
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_single_outcome",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)
                required_keys = [
                    "completionId",
                    "userId",
                    "timestamp",
                    "prompt",
                    "completion",
                    "model",
                    "version",
                ]

                # if any of the required keys are missing, return 400
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_single_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                try:
                    privacy = PrivacySetting(data.get("privacy"))
                except ValueError:
                    error_msg = f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_single_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                if data["model"] == "test":
                    return {"status": "success"}

                data["versionBackend"] = self.version_backend
                self.amplitude.track(
                    BaseEvent(
                        event_type="add_single_outcome",
                        user_id=data["userId"],
                        event_properties={"model": data["model"]},
                    )
                )

                self.firebase_client.upload_data(collection, data, privacy)
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/add_single_outcome",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        @self.app.put("/add_completion_outcome")
        @self.limiter.limit("1000/minute")
        async def add_completion_outcome(
            request: Request, background_tasks: BackgroundTasks
        ):
            try:
                collection = self.settings[self.FIREBASE_COLLECTIONS_KEY]["outcomes"]
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion_outcome",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)
                required_keys = [
                    "pairId",
                    "userId",
                    "acceptedIndex",
                    "version",
                    "completionItems",
                ]
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                data["versionBackend"] = self.version_backend
                data["timestamp"] = int(datetime.now().timestamp())

                required_completion_keys = [
                    "completionId",
                    "prompt",
                    "completion",
                    "model",
                ]

                completion_items = data["completionItems"]
                if len(completion_items) != 2:
                    error_msg = "completionItems must contain 2 items"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                for completion_item in completion_items:
                    if not all(
                        key in completion_item for key in required_completion_keys
                    ):
                        ERROR_COUNT.labels(
                            method="POST",
                            endpoint="/add_completion_outcome",
                            exception="BadRequest",
                            status=400,
                        ).inc()
                        raise HTTPException(
                            status_code=400,
                            detail=f"completion_item must contain {required_completion_keys}",
                        )

                try:
                    privacy = PrivacySetting(data.get("privacy"))
                except ValueError:
                    error_msg = f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                if completion_items[0]["model"] == "test":
                    return {"status": "success"}

                background_tasks.add_task(
                    self.amplitude.track,
                    BaseEvent(
                        event_type="add_completion_outcome",
                        user_id=data["userId"],
                        event_properties={
                            "model1": completion_items[0]["model"],
                            "model2": completion_items[1]["model"],
                        },
                    ),
                )
                background_tasks.add_task(
                    self.firebase_client.upload_data, collection, data, privacy
                )
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/add_completion_outcome",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        @self.app.put("/add_edit_outcome")
        @self.limiter.limit("1000/minute")
        async def add_edit_outcome(request: Request, background_tasks: BackgroundTasks):
            try:
                collection = self.settings[self.FIREBASE_COLLECTIONS_KEY][
                    "edit_outcomes"
                ]
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_completion_outcome",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)
                required_keys = [
                    "pairId",
                    "userId",
                    "acceptedIndex",
                    "version",
                    "responseItems",
                ]
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_edit_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                data["versionBackend"] = self.version_backend
                data["timestamp"] = int(datetime.now().timestamp())

                required_response_keys = [
                    "responseId",
                    "prompt",
                    "response",
                    "model",
                ]

                response_items = data["responseItems"]
                if len(response_items) != 2:
                    error_msg = "responseItems must contain 2 items"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_edit_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                for response_item in response_items:
                    if not all(key in response_item for key in required_response_keys):
                        ERROR_COUNT.labels(
                            method="POST",
                            endpoint="/add_edit_outcome",
                            exception="BadRequest",
                            status=400,
                        ).inc()
                        raise HTTPException(
                            status_code=400,
                            detail=f"response_item must contain {required_response_keys}",
                        )

                try:
                    privacy = PrivacySetting(data.get("privacy"))
                except ValueError:
                    error_msg = f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/add_edit_outcome",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                if response_items[0]["model"] == "test":
                    return {"status": "success"}

                background_tasks.add_task(
                    self.amplitude.track,
                    BaseEvent(
                        event_type="add_edit_outcome",
                        user_id=data["userId"],
                        event_properties={
                            "model1": response_items[0]["model"],
                            "model2": response_items[1]["model"],
                        },
                    ),
                )
                background_tasks.add_task(
                    self.firebase_client.upload_data, collection, data, privacy
                )
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/add_edit_outcome",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        @self.app.get("/list_models")
        @self.limiter.limit("1000/minute")
        async def list_models(request: Request):
            return {"models": list(self.models)}

        @self.app.post("/user_scores")
        @self.limiter.limit("1000/minute")
        async def get_user_scores(request: Request):
            try:
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/user_scores",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                required_keys = [
                    "userId",
                ]
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/user_scores",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                user_id = data.get("userId")

                autocomplete_outcomes_collection_name = self.settings[
                    self.FIREBASE_COLLECTIONS_KEY
                ]["outcomes"]

                start_time = time.time()
                logger.info("Personal Ranking User Id: {}".format(user_id))
                outcomes_df = self.firebase_client.get_autocomplete_outcomes(
                    autocomplete_outcomes_collection_name, user_id=user_id
                )
                end_time = time.time()
                logger.info(
                    f"Time taken to retrieve data for personal_ranking: {end_time - start_time:.2f} seconds"
                )

                start_time = time.time()
                scores_over_time = get_scores(
                    self.global_outcomes_df,
                    outcomes_df,
                    models=self.models,
                    interval_size=20,
                )
                end_time = time.time()
                logger.info(
                    f"Time taken to calculate personal ranking: {end_time - start_time:.2f} seconds"
                )

                response = JSONResponse(content=scores_over_time)
                return response
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/user_scores",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        @self.app.post("/user_vote_count")
        @self.limiter.limit("1000/minute")
        async def get_user_vote_count(request: Request):
            try:
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/user_vote_count",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                required_keys = [
                    "userId",
                ]
                if not all(key in data for key in required_keys):
                    error_msg = f"Request must contain {required_keys}"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/user_vote_count",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                user_id = data.get("userId")
                user_vote_count = self.firebase_client.get_autocomplete_outcomes_count(
                    self.settings[self.FIREBASE_COLLECTIONS_KEY]["outcomes"], user_id
                )
                response = JSONResponse(content={"voteCount": user_vote_count})
                return response
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/user_vote_count",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(
                    status_code=500, detail="An internal server error occurred."
                )

        """
        USER ROUTES

        """

        @self.app.put("/users")
        async def add_user(user: User):
            """
            Add a new user
            """
            try:
                self.user_repo.create_user(
                    user.user_id,
                    user.username,
                    user.password,
                    user.metadata.dict() if user.metadata else None,
                )
                return {
                    "message": "User created successfully",
                    "username": user.username,
                }
            except UsernameExistsError as e:
                raise HTTPException(status_code=409, detail=str(e))

        @self.app.patch("/users")
        async def update_user(user: UpdateUser):
            """
            Update the user
            """
            try:
                self.user_repo.update_user(
                    user.username,
                    user.old_password,
                    user.new_password,
                    user.metadata.dict() if user.metadata else None,
                )
                return {"message": "User updated successfully"}
            except UserNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except PasswordIncorrectError as e:
                raise HTTPException(status_code=401, detail=str(e))

        @self.app.post("/users/authenticate")
        async def authenticate_user(user: User):
            """
            Authenticate the user and return 200 if user and password match
            """
            try:
                authenticated_user = self.user_repo.authenticate(
                    user.username, user.user_id, user.password
                )
                return {
                    "message": "Authentication successful",
                    "user": authenticated_user,
                }
            except (UserNotFoundError, PasswordIncorrectError) as e:
                raise HTTPException(
                    status_code=401, detail="Invalid username or password"
                )

        # The rest of these are for testing
        @self.app.post("/create_completion")
        @self.limiter.limit("1000/minute")
        async def create_completion(request: Request):
            """
            This route is only used for evaluation purposes
            """
            try:
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_completion",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                if "prefix" not in data or "userId" not in data or "model" not in data:
                    error_msg = "Request must contain prefix, userId, and model"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_completion",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                # Extract data from the request
                prefix = data.get("prefix")
                user_id = data.get("userId")
                model = data.get("model")

                # Optional parameters with default values
                completion_id = data.get("completionId", str(uuid.uuid4()))
                temperature = data.get("temperature", TEMPERATURE)
                max_tokens = data.get("max_tokens", data.get("maxTokens", MAX_TOKENS))
                top_p = data.get("top_p", data.get("topP", TOP_P))
                max_lines = data.get("max_lines", data.get("maxLines", 1000))
                suffix = data.get("suffix", "")

                prompt_index = data.get(
                    "prompt_index", PROMPT_INDEX_FIM
                )  # Use the new constant as default

                # Create state and options
                state = State(prefix=prefix, suffix=suffix)
                options = LLMOptions(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    max_lines=max_lines,
                    prompt_index=prompt_index,
                )

                # Get the client for the specified model
                client: IBaseClient = self.model_clients.get(model)

                # Use the timed create
                completion, latency = await timed_create(
                    client=client, state=state, model=model, options=options
                )

                # Create the completion item
                completion_item = {
                    "completionId": completion_id,
                    "userId": user_id,
                    "timestamp": int(datetime.now().timestamp()),
                    "prompt": client.generate_prompt_for_model(
                        state=state, model=model, prompt_index=prompt_index
                    ),
                    "prefix": state.prefix,
                    "suffix": state.suffix,
                    "latency": latency,
                    "completion": completion,
                    "model": model,
                }

                # Return the response
                logging.info(completion_item)
                return JSONResponse(content=completion_item)

            except Exception as e:
                # Log the error and return an appropriate error response
                logging.error(f"Error in create_completion endpoint: {str(e)}")
                ERROR_COUNT.labels(
                    method="POST",
                    endpoint="/create_completion",
                    exception="ServerError",
                    status=500,
                ).inc()
                raise HTTPException(status_code=500, detail="Internal server error")

        # Used exclusively for load testing
        @self.app.post("/create_pair_test")
        @self.limiter.limit("300/minute")
        async def create_pair_test(request: Request, background_tasks: BackgroundTasks):
            try:
                start_time = time.time()
                latency_breakdown = {}

                # Request parsing
                request_parse_start = time.time()
                try:
                    data = await request.json()
                except JSONDecodeError:
                    error_msg = "Invalid JSON in request body"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_pair_test",
                        exception="JSONDecodeError",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                if (
                    "prefix" not in data
                    or "userId" not in data
                    or "privacy" not in data
                ):
                    error_msg = "Request must contain keys: [prefix, userId, privacy]"
                    logger.info(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_pair_test",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(status_code=400, detail=error_msg)

                try:
                    privacy = PrivacySetting(data.get("privacy"))
                except ValueError:
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_pair_test",
                        exception="BadRequest",
                        status=400,
                    ).inc()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid privacy value. Must be one of: {', '.join(PrivacySetting)}",
                    )

                latency_breakdown["request_parsing"] = time.time() - request_parse_start

                # Data preparation
                data_prep_start = time.time()
                pairId = data.get("pairId", str(uuid.uuid4()))
                temperature = data.get("temperature", TEMPERATURE)
                max_tokens = data.get("max_tokens", data.get("maxTokens", MAX_TOKENS))
                top_p = data.get("top_p", data.get("topP", TOP_P))
                max_lines = data.get("max_lines", data.get("maxLines", MAX_LINES))
                prefix = data.get("prefix")
                suffix = data.get("suffix", "")
                user_id = data.get("userId")
                tags = data.get("modelTags", [])

                state = State(prefix=prefix, suffix=suffix)
                options = LLMOptions(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    max_lines=max_lines,
                )
                latency_breakdown["data_preparation"] = time.time() - data_prep_start

                # Model selection
                model_selection_start = time.time()

                tagged_models = get_models_by_tags(
                    tags, self.models, self.tag_to_models
                )
                if len(tagged_models) == 0:
                    tagged_models = self.models
                tagged_model_indices = [
                    self.models.index(model) for model in tagged_models
                ]
                model_probabilities = self.model_probabilities[tagged_model_indices]
                model_probabilities /= model_probabilities.sum()

                selected_models = np.random.choice(
                    tagged_models, size=2, replace=False, p=model_probabilities
                ).tolist()
                client1: IBaseClient = self.model_clients.get(selected_models[0])
                client2: IBaseClient = self.model_clients.get(selected_models[1])
                latency_breakdown["model_selection"] = (
                    time.time() - model_selection_start
                )

                # Concurrent task execution
                task_execution_start = time.time()
                task1 = asyncio.create_task(
                    timed_create_test(
                        client=client1,
                        state=state,
                        model=selected_models[0],
                        options=options,
                    )
                )
                task2 = asyncio.create_task(
                    timed_create_test(
                        client=client2,
                        state=state,
                        model=selected_models[1],
                        options=options,
                    )
                )
                (completion1, latency1), (completion2, latency2) = await asyncio.gather(
                    task1, task2
                )
                latency_breakdown["task_execution"] = time.time() - task_execution_start
                latency_breakdown["model1_latency"] = latency1
                latency_breakdown["model2_latency"] = latency2

                # Response preparation
                response_prep_start = time.time()
                completionId1, completionId2 = str(uuid.uuid4()), str(uuid.uuid4())
                timestamp = int(datetime.now().timestamp())
                prompt1 = client1.generate_prompt_for_model(
                    state=state, model=selected_models[0], prompt_index=0
                )
                prompt2 = client2.generate_prompt_for_model(
                    state=state, model=selected_models[1], prompt_index=0
                )

                completionItem1 = {
                    "completionId": completionId1,
                    "userId": user_id,
                    "timestamp": timestamp,
                    "prompt": prompt1,
                    "prefix": state.prefix,
                    "suffix": state.suffix,
                    "prompt_length": len(prompt1),
                    "prefix_length": len(state.prefix),
                    "suffix_length": len(state.suffix),
                    "latency": latency1,
                    "completion": completion1,
                    "model": selected_models[0],
                    "pairCompletionId": completionId2,
                    "pairIndex": 0,
                    "versionBackend": self.version_backend,
                }
                completionItem2 = {
                    "completionId": completionId2,
                    "userId": user_id,
                    "timestamp": timestamp,
                    "prompt": prompt2,
                    "prefix": state.prefix,
                    "suffix": state.suffix,
                    "prompt_length": len(prompt2),
                    "prefix_length": len(state.prefix),
                    "suffix_length": len(state.suffix),
                    "latency": latency2,
                    "completion": completion2,
                    "model": selected_models[1],
                    "pairCompletionId": completionId1,
                    "pairIndex": 1,
                    "versionBackend": self.version_backend,
                }

                json_content = {
                    "pairId": pairId,
                    "completionItems": [completionItem1, completionItem2],
                }
                privacy_aware_log(json_content, privacy, logger, logging.INFO)
                latency_breakdown["response_preparation"] = (
                    time.time() - response_prep_start
                )

                # Firebase upload
                firebase_upload_start = time.time()
                collection = "load_test"
                background_tasks.add_task(
                    self.firebase_client.upload_data,
                    collection,
                    completionItem1,
                    privacy,
                )
                background_tasks.add_task(
                    self.firebase_client.upload_data,
                    collection,
                    completionItem2,
                    privacy,
                )
                latency_breakdown["firebase_upload"] = (
                    time.time() - firebase_upload_start
                )

                # Final response
                response = JSONResponse(content=json_content)
                end_time = time.time()
                total_latency = end_time - start_time
                latency_breakdown["total_latency"] = total_latency

                logger.info(latency_breakdown)
                logger.info(
                    f"Total Latency: {total_latency:.4f}s, model {selected_models[0]}: {latency1:.4f}s, model {selected_models[1]}: {latency2:.4f}s | {completionId1} | {completionId2}"
                )

                return response
            except HTTPException:
                raise
            except Exception as e:
                # TODO Handle errors specific to each API client.
                if "429" in str(e):
                    logger.error(
                        "Rate limit exceeded. selected_models: {}".format(
                            selected_models
                        )
                    )
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_pair_test",
                        exception="RateLimitExceeded",
                        status=429,
                    ).inc()
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please try again later.",
                    )
                else:
                    error_msg = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    ERROR_COUNT.labels(
                        method="POST",
                        endpoint="/create_pair_test",
                        exception="ServerError",
                        status=500,
                    ).inc()
                    raise HTTPException(
                        status_code=500, detail="An internal server error occurred."
                    )

        @self.app.get("/metrics")
        async def metrics(request: Request):
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            method = request.method
            path = request.url.path
            start_time = time.time()

            try:
                response = await call_next(request)
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(latency)
                REQUEST_COUNT.labels(
                    method=method, endpoint=path, status=response.status_code
                ).inc()

                return response
            except Exception as e:
                raise


fastapp = FastAPIApp()
app = fastapp.app
