import pytest
from fastapi.testclient import TestClient
from apis.base_client import LLMResponse
from unittest.mock import patch
from app import app
from src.privacy import PrivacySetting

client = TestClient(app)

all_models = {
    "gpt-4o-mini-2024-07-18",
    "claude-3-haiku-20240307",
    "llama-3.1-70b-instruct",
    "codestral-2405",
    "deepseek-coder-fim",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "claude-3-5-sonnet-20240620",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest",
}

fast_fim_models = {
    "gpt-4o-mini-2024-07-18",  # fast
    "claude-3-haiku-20240307",  # fast
    "codestral-2405",  # fast, fim
    "deepseek-coder-fim",  # fim
    "gemini-1.5-flash-001",  # fast
}


@pytest.fixture
def mock_firebase_upload():
    with patch("app.FirebaseClient.upload_data") as mock:
        yield mock


@pytest.fixture
def mock_timed_create():
    async def mock_timed_create_impl(client, state, model, options):
        # You can customize these return values as needed
        result = LLMResponse(
            text=f"Mocked completion for {model}", raw_text=f"raw_text"
        )
        latency = 0.5
        return result, latency

    with patch("app.timed_create", side_effect=mock_timed_create_impl) as mock:
        yield mock


def test_create_pair_happy_path(mock_firebase_upload, mock_timed_create):
    response = client.post(
        "/create_pair",
        json={
            "prefix": "Test prefix",
            "userId": "test_user",
            "privacy": PrivacySetting.RESEARCH.value,
            "pairId": "test_pair_id",
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.8,
            "max_lines": 10,
            "suffix": "Test suffix",
        },
    )
    assert response.status_code == 200
    assert "pairId" in response.json()
    assert "completionItems" in response.json()
    assert mock_timed_create.call_count == 2  # timed_create should be called twice


def test_create_edit_pair_happy_path(mock_firebase_upload, mock_timed_create):
    response = client.post(
        "/create_edit_pair",
        json={
            "prefix": "Test prefix",
            "userId": "test_user",
            "codeToEdit": "Test highlighted",
            "userInput": "test input",
            "language": "python",
            "privacy": PrivacySetting.RESEARCH.value,
            "pairId": "test_pair_id",
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.8,
            "max_lines": 10,
            "suffix": "Test suffix",
        },
    )
    assert response.status_code == 200
    assert "pairId" in response.json()
    assert "responseItems" in response.json()
    assert mock_firebase_upload.call_count == 2  # Two completions should be uploaded
    assert mock_timed_create.call_count == 2  # timed_create should be called twice


def test_create_pair_missing_keys():
    response = client.post("/create_pair", json={"prefix": "Test prefix"})
    assert response.status_code == 400
    assert "must contain keys: [prefix, userId, privacy]" in response.json()["detail"]


def test_create_pair_missing_privacy():
    response = client.post(
        "/create_pair", json={"prefix": "Test prefix", "userId": "test_user"}
    )
    assert response.status_code == 400
    assert "must contain keys: [prefix, userId, privacy]" in response.json()["detail"]


def test_add_completion_happy_path(mock_firebase_upload):
    response = client.put(
        "/add_completion",
        json={
            "completionId": "test_completion_id",
            "pairCompletionId": "test_pair_completion_id",
            "pairIndex": 0,
            "userId": "test_user",
            "timestamp": 1628000000,
            "prompt": "Test prompt",
            "completion": "Test completion",
            "model": "model1",
            "version": "1.0",
            "privacy": PrivacySetting.DEBUG.value,
        },
    )
    assert response.status_code == 200
    mock_firebase_upload.assert_called_once()


def test_add_completion_missing_keys():
    response = client.put(
        "/add_completion", json={"completionId": "test_completion_id"}
    )
    assert response.status_code == 400
    assert "Request must contain" in response.json()["detail"]


def test_add_completion_incorrect_privacy():
    response = client.put(
        "/add_completion",
        json={
            "completionId": "test_completion_id",
            "pairCompletionId": "test_pair_completion_id",
            "pairIndex": 0,
            "userId": "test_user",
            "timestamp": 1628000000,
            "prompt": "Test prompt",
            "completion": "Test completion",
            "model": "model1",
            "version": "1.0",
            "privacy": "wrong",
        },
    )
    assert response.status_code == 400
    assert "Invalid privacy value" in response.json()["detail"]


def test_add_single_outcome_happy_path(mock_firebase_upload):
    response = client.put(
        "/add_single_outcome",
        json={
            "completionId": "test_completion_id",
            "userId": "test_user",
            "timestamp": 1628000000,
            "prompt": "Test prompt",
            "completion": "Test completion",
            "model": "model1",
            "version": "1.0",
            "privacy": PrivacySetting.PRIVATE.value,
        },
    )
    assert response.status_code == 200
    mock_firebase_upload.assert_called_once()


def test_add_single_outcome_missing_keys():
    response = client.put(
        "/add_single_outcome", json={"completionId": "test_completion_id"}
    )
    assert response.status_code == 400
    assert "Request must contain" in response.json()["detail"]


def test_add_single_outcome_missing_privacy():
    response = client.put(
        "/add_single_outcome",
        json={
            "completionId": "test_completion_id",
            "userId": "test_user",
            "timestamp": 1628000000,
            "prompt": "Test prompt",
            "completion": "Test completion",
            "model": "model1",
            "version": "1.0",
        },
    )
    assert response.status_code == 400
    assert "Invalid privacy value" in response.json()["detail"]


def test_add_completion_outcome_happy_path(mock_firebase_upload):
    response = client.put(
        "/add_completion_outcome",
        json={
            "pairId": "test_pair_id",
            "userId": "test_user",
            "acceptedIndex": 0,
            "version": "1.0",
            "completionItems": [
                {
                    "completionId": "test_completion_id_1",
                    "prompt": "Test prompt 1",
                    "completion": "Test completion 1",
                    "model": "model1",
                },
                {
                    "completionId": "test_completion_id_2",
                    "prompt": "Test prompt 2",
                    "completion": "Test completion 2",
                    "model": "model2",
                },
            ],
            "privacy": PrivacySetting.RESEARCH.value,
        },
    )
    assert response.status_code == 200
    mock_firebase_upload.assert_called_once()


def test_add_completion_outcome_missing_keys():
    response = client.put("/add_completion_outcome", json={"pairId": "test_pair_id"})
    assert response.status_code == 400
    assert "Request must contain" in response.json()["detail"]


def test_add_completion_outcome_incorrect_privacy():
    response = client.put(
        "/add_completion_outcome",
        json={
            "pairId": "test_pair_id",
            "userId": "test_user",
            "acceptedIndex": 0,
            "version": "1.0",
            "completionItems": [
                {
                    "completionId": "test_completion_id_1",
                    "prompt": "Test prompt 1",
                    "completion": "Test completion 1",
                    "model": "model1",
                },
                {
                    "completionId": "test_completion_id_2",
                    "prompt": "Test prompt 2",
                    "completion": "Test completion 2",
                    "model": "model2",
                },
            ],
            "privacy": "wrong",
        },
    )
    assert response.status_code == 400
    assert "Invalid privacy value" in response.json()["detail"]


def test_list_models():
    response = client.get("/list_models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert isinstance(response.json()["models"], list)
