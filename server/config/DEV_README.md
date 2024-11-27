This contains instructions on how to setup the project.

# Setup

## Config Files
Some files are not uploaded to git, but must be added to the config folder
```
app_config.yaml # Contains general app configuration
api_config.py  # Contains LLM API keys
amplitude_config.py # Contains amplitude API Key
firebase_config.json # Contains firebase configuration
prometheus.yml # Contains the prometheus target
```

Amplitude, Firebase, and Fly use the normal configurations from those services.

Below are what the app, api, and prometheus config files look like in case you wanted to set it up on your own.

### App Config

```
models:
  gpt-4o-mini-2024-07-18:
    weight: 0.1
    tags: [edit]  # Any models with edit should be tagged with edit
    input_cost: 0.15
    output_cost: 0.6
  # More models

# Firebase collections
firebase_collections:
  all_completions: firebase_path
  completions: firebase_path
  single_outcomes: firebase_path
  outcomes: firebase_path
  edits: firebase_path
  edit_outcomes: firebase_path

# API version
version_backend: x.x.x
```

### API Config
```
PROVIDER_API_KEY="YOUR_API_KEY"
```

### Prometheus Config
```
- targets:
  - {HOST_ADDRESS} 
  labels:
    env: production
```

## Testing

1. Run Pytest and ensure all tests pass

```
pytest
```

## Deploying

Make sure to follow steps in Testing first

### Local

To build:
```
docker compose up --build
```

To remove:
```
docker compose down --remove-orphans
```

### Fly

We use Fly.io as our hosting service. If the fly.toml file doesn't exist, it will be created.

1. Change version number in `config/app_config.yaml`
2. Run the command
```
fly deploy
```
