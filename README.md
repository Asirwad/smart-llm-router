# Smart Model Router

**Cost-Control Smart Model Router** — Intelligently routes prompts to the cheapest capable model.

## 🎯 Overview

A single API endpoint that automatically routes each prompt to the most cost-effective LLM based on complexity:

| Complexity | Model | Location |
|------------|-------|----------|
| Simple | Granite 4.0 Nano (350M) | Local (Ollama) |
| Medium | Gemini 2.0 Flash | GCP API |
| Complex | Gemini 1.5 Pro | GCP API |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- NVIDIA GPU with Docker support (for local Ollama)
- Google Cloud API key for Gemini

### Setup

1. **Clone and configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

2. **Start infrastructure**
   ```bash
   docker-compose up -d postgres redis ollama
   ```

3. **Pull the local model**
   ```bash
   ollama pull granite4:350m
   ```

4. **Install Python dependencies (using uv)**
   ```bash
   uv pip install -e ".[dev]"
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the API server**
   ```bash
   uvicorn src.main:app --reload
   ```

7. **Access the API**
   - API Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## 📁 Project Structure

```
smart-model-router/
├── src/
│   ├── api/           # HTTP layer (routes, schemas, auth)
│   ├── core/          # Business logic (router, cost calculator)
│   ├── providers/     # LLM integrations (Ollama, Gemini)
│   ├── db/            # Database models and sessions
│   ├── services/      # Cross-cutting (caching, logging)
│   ├── config.py      # Configuration management
│   └── main.py        # FastAPI application
├── tests/             # Test suite
├── alembic/           # Database migrations
├── docker-compose.yml # Full stack orchestration
└── Dockerfile         # Application container
```

## 🔧 API Usage

```bash
# Make a request
curl -X POST http://localhost:8000/v1/complete \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize this text"}'
```

## 📊 Response Format

```json
{
  "response": "...",
  "model_used": "granite4:350m",
  "difficulty_tag": "simple",
  "estimated_cost": 0.0,
  "estimated_savings": 0.015
}
```

## 🧪 Development

```bash
# Run tests
pytest

# Format code
ruff format .

# Lint
ruff check .
```

## 📝 License

MIT
