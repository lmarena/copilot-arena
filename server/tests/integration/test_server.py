import pytest
import requests
import time
import subprocess
import os
from typing import Generator
import docker
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now we can import from app.py
from app import app


class DockerComposeManager:
    def __init__(self):
        self.project_root = project_root
        self.compose_file = self.project_root / "docker-compose.yml"
        self.docker_client = docker.from_env()

    def start_services(self) -> None:
        """Start Docker Compose services with build."""
        print("Starting Docker Compose services with build...")
        result = subprocess.run(
            ["docker", "compose", "-f", str(self.compose_file), "up", "--build", "-d"],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        if result.returncode != 0:
            raise Exception(f"Docker Compose up failed: {result.stderr}")
        print(result.stdout)

    def stop_services(self) -> None:
        """Stop and remove Docker Compose services."""
        print("Stopping Docker Compose services...")
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            capture_output=True,
            cwd=str(self.project_root),
        )

        # Clean up any leftover containers
        for container in self.docker_client.containers.list(all=True):
            if container.name in ["copilot_arena_app", "copilot_nginx"]:
                try:
                    container.remove(force=True)
                except docker.errors.APIError:
                    print(f"Failed to remove container {container.name}")

    def get_logs(self) -> str:
        """Get logs from all containers."""
        logs = ""
        for container in self.docker_client.containers.list():
            if container.name in ["copilot_arena_app", "copilot_nginx"]:
                logs += f"\n=== Logs for {container.name} ===\n"
                logs += container.logs().decode("utf-8")
        return logs


def wait_for_service(url: str, max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for the service to become available."""
    print(f"Waiting for service at {url}...")
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            print(f"Attempt {attempt + 1}: Status code {response.status_code}")

            if (
                response.status_code != 502
            ):  # Any response except 502 means nginx is connected to backend
                print(f"Service is up! Got status code {response.status_code}")
                return True

            print(f"Got 502, backend might not be ready. Waiting {delay} seconds...")
            time.sleep(delay)

        except requests.ConnectionError:
            print(
                f"Attempt {attempt + 1}/{max_retries} failed, retrying in {delay} seconds..."
            )
            time.sleep(delay)

    print("Service never became available after all retries")
    return False


@pytest.fixture(scope="session")
def docker_compose() -> Generator[DockerComposeManager, None, None]:
    """Fixture to handle Docker Compose lifecycle."""
    # Verify required files exist
    required_files = [
        project_root / "docker-compose.yml",
        project_root / "Dockerfile",
        project_root / "app.py",
    ]
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file {file} not found")

    # Initialize and start services
    docker_manager = DockerComposeManager()
    try:
        # Clean up any existing services
        docker_manager.stop_services()

        # Start services with build
        docker_manager.start_services()

        # Wait for service to be ready
        base_url = "http://localhost:8080"
        if not wait_for_service(base_url):
            logs = docker_manager.get_logs()
            raise Exception(f"Service failed to start. Docker logs:\n{logs}")

        yield docker_manager

    except Exception as e:
        logs = docker_manager.get_logs()
        print(f"Setup failed. Docker logs:\n{logs}")
        raise e

    finally:
        # Cleanup
        docker_manager.stop_services()


def test_create_pair(docker_compose):
    """Test the create_pair endpoint with sample data."""
    url = "http://localhost:8080/create_pair"

    payload = {
        "prefix": "print('hello",
        "userId": "test",
        "privacy": "Private",
        "modelTags": [],
    }

    response = requests.post(url, json=payload)

    # Log response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    # Assertions
    assert (
        response.status_code == 200
    ), f"Expected status code 200, got {response.status_code}"

    response_data = response.json()
    assert "pairId" in response_data, "Response should contain an id"
    assert "completionItems" in response_data, "Response should contain completionItems"


def test_create_edit_pair(docker_compose):
    """Test the create_pair endpoint with sample data."""
    url = "http://localhost:8080/create_edit_pair"

    payload = {
        "prefix": 'def transform_df(df):\n    new_df = pd.DataFrame(\n        columns=["age", "blue", "brown", "green", "month", "day", "height"]\n    )\n    new_df["age"] = df["age"]\n    new_df["weight"] = df["weight"] * 100\n',
        "codeToEdit": '    # Convert \'height\' from meters to centimeters\n    new_df["height"] = df["height"] * 100\n',
        "suffix": '    new_df["month"] = df["month"]',
        "userInput": "change height to weight",
        "language": "python",
        "userId": "test",
        "privacy": "Private",
        "modelTags": [],
    }

    response = requests.post(url, json=payload)

    # Log response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    # Assertions
    assert (
        response.status_code == 200
    ), f"Expected status code 200, got {response.status_code}"

    response_data = response.json()
    assert "pairId" in response_data, "Response should contain an id"
    assert "responseItems" in response_data, "Response should contain responseItems"


def test_user_scores(docker_compose):
    """Test the user_scores endpoint."""
    url = "http://localhost:8080/user_scores"

    payload = {"userId": "test_user"}

    response = requests.post(url, json=payload)

    # Log response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    # Assertions
    assert (
        response.status_code == 200
    ), f"Expected status code 200, got {response.status_code}"

    response_data = response.json()
    # The response should be a dictionary containing model scores over time
    assert isinstance(response_data, list), "Response should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
