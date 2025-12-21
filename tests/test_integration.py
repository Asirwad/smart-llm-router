"""
End-to-End Integration Tests for Smart Model Router.

Tests the complete flow from API request to response,
validating all components work together correctly.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest_asyncio.fixture
async def client():
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def api_key(client):
    """Create a test API key and return the raw key."""
    response = await client.post(
        "/v1/keys",
        json={"name": "Integration Test Key"},
    )
    assert response.status_code == 201
    data = response.json()
    return data["key"]


@pytest_asyncio.fixture
async def clear_cache():
    """Clear Redis cache before/after tests."""
    from src.services import get_cache_service
    cache = get_cache_service()
    await cache.clear_all()
    yield
    await cache.clear_all()


class TestHealthEndpoints:
    """Test health and root endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Root returns API info."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Smart Model Router"
        assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Health check returns status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data


class TestAPIKeyManagement:
    """Test API key CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_api_key(self, client):
        """Create new API key."""
        response = await client.post(
            "/v1/keys",
            json={"name": "Test Key"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Key"
        assert data["key"].startswith("smr_")
        assert data["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_list_api_keys(self, client, api_key):
        """List existing API keys."""
        response = await client.get("/v1/keys")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert data["total"] >= 1
        # Raw keys should NOT be in list
        for key in data["keys"]:
            assert key["key"] is None
    
    @pytest.mark.asyncio
    async def test_deactivate_api_key(self, client):
        """Deactivate an API key."""
        # Create key
        create_resp = await client.post(
            "/v1/keys",
            json={"name": "To Deactivate"},
        )
        key_id = create_resp.json()["id"]
        
        # Deactivate
        response = await client.delete(f"/v1/keys/{key_id}")
        assert response.status_code == 204


class TestAuthentication:
    """Test API key authentication."""
    
    @pytest.mark.asyncio
    async def test_missing_api_key(self, client):
        """Request without API key returns 401."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self, client):
        """Request with invalid API key returns 401."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Hello"},
            headers={"X-API-Key": "invalid_key"},
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_valid_api_key(self, client, api_key):
        """Request with valid API key succeeds."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "What is 2+2?"},
            headers={"X-API-Key": api_key},
        )
        # Should succeed (200) or service unavailable (503) if LLM is down
        assert response.status_code in [200, 503]


class TestCompletionEndpoint:
    """Test the main completion endpoint."""
    
    @pytest.mark.asyncio
    async def test_simple_prompt(self, client, api_key):
        """Simple prompt returns valid response."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Summarize: Python is a programming language."},
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "model_used" in data
            assert "difficulty_tag" in data
            assert "estimated_cost" in data
            assert "latency_ms" in data
    
    @pytest.mark.asyncio
    async def test_force_tier_simple(self, client, api_key):
        """Force tier overrides routing."""
        response = await client.post(
            "/v1/complete",
            json={
                "prompt": "Design a complex architecture",
                "force_tier": "simple",
            },
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["difficulty_tag"] == "simple"
    
    @pytest.mark.asyncio
    async def test_force_tier_complex(self, client, api_key):
        """Force tier to complex."""
        response = await client.post(
            "/v1/complete",
            json={
                "prompt": "What is 2+2?",
                "force_tier": "complex",
            },
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["difficulty_tag"] == "complex"


class TestCaching:
    """Test Redis caching behavior."""
    
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, client, api_key, clear_cache):
        """First request is cache miss, second is hit."""
        prompt = "What is caching? (test unique prompt)"
        headers = {"X-API-Key": api_key}
        
        # First request - cache miss
        r1 = await client.post(
            "/v1/complete",
            json={"prompt": prompt},
            headers=headers,
        )
        if r1.status_code != 200:
            pytest.skip("LLM not available")
        
        data1 = r1.json()
        assert data1["cache_hit"] is False
        latency1 = data1["latency_ms"]
        
        # Second request - cache hit
        r2 = await client.post(
            "/v1/complete",
            json={"prompt": prompt},
            headers=headers,
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["cache_hit"] is True
        latency2 = data2["latency_ms"]
        
        # Cached response should be much faster
        assert latency2 < latency1, f"Cache hit ({latency2}ms) should be faster than miss ({latency1}ms)"
    
    @pytest.mark.asyncio
    async def test_different_prompts_not_cached(self, client, api_key, clear_cache):
        """Different prompts get different cache entries."""
        headers = {"X-API-Key": api_key}
        
        r1 = await client.post(
            "/v1/complete",
            json={"prompt": "Unique prompt 1"},
            headers=headers,
        )
        r2 = await client.post(
            "/v1/complete",
            json={"prompt": "Unique prompt 2"},
            headers=headers,
        )
        
        if r1.status_code == 200 and r2.status_code == 200:
            assert r1.json()["cache_hit"] is False
            assert r2.json()["cache_hit"] is False


class TestCostTracking:
    """Test cost calculation and logging."""
    
    @pytest.mark.asyncio
    async def test_cost_fields_present(self, client, api_key):
        """Response includes cost fields."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Hello"},
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            assert "estimated_cost" in data
            assert "estimated_savings" in data
            assert isinstance(data["estimated_cost"], (int, float))
            assert isinstance(data["estimated_savings"], (int, float))
    
    @pytest.mark.asyncio
    async def test_local_model_saves_money(self, client, api_key, clear_cache):
        """Simple prompts using local model should show savings."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Summarize: Test"},
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            # Local model should have savings vs Pro baseline
            if "granite" in data["model_used"].lower():
                assert data["estimated_savings"] > 0


class TestInputValidation:
    """Test request validation."""
    
    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self, client, api_key):
        """Empty prompt is rejected."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": ""},
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_invalid_force_tier(self, client, api_key):
        """Invalid force_tier is rejected."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Hello", "force_tier": "invalid"},
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 422


class TestRouterClassification:
    """Test prompt classification/routing."""
    
    @pytest.mark.asyncio
    async def test_simple_prompt_classification(self, client, api_key, clear_cache):
        """Simple prompts get simple/medium tier."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Translate to Spanish: Hello"},
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            # Translation is a simple task
            assert data["difficulty_tag"] in ["simple", "medium"]
    
    @pytest.mark.asyncio
    async def test_best_model_override(self, client, api_key, clear_cache):
        """'best model' keyword triggers complex tier."""
        response = await client.post(
            "/v1/complete",
            json={"prompt": "Use the best model to answer: What is 2+2?"},
            headers={"X-API-Key": api_key},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["difficulty_tag"] == "complex"
