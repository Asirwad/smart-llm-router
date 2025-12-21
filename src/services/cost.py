"""
Cost Calculator Service.

Handles:
- Per-model pricing configuration
- Cost estimation from token counts
- Savings calculation vs baseline (always using Pro model)
- Request logging to database
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.router import DifficultyTier
from src.db import RequestLog
from src.providers.base import ProviderResponse


@dataclass
class ModelPricing:
    """
    Pricing information for a model.
    
    Costs are in USD per 1M tokens (standard industry unit).
    """
    model_name: str
    input_cost_per_million: float   # Cost per 1M input tokens
    output_cost_per_million: float  # Cost per 1M output tokens
    is_local: bool = False          # Local models have ~zero marginal cost


# ===================
# Model Pricing Table
# ===================
# Prices based on public API pricing (as of Dec 2024)
# Local models have electricity/GPU cost approximated

MODEL_PRICING: dict[str, ModelPricing] = {
    # Local Ollama Models (minimal cost - just electricity/GPU time)
    "granite4:350m": ModelPricing(
        model_name="granite4:350m",
        input_cost_per_million=0.01,   # ~$0.01 per 1M tokens (electricity)
        output_cost_per_million=0.01,
        is_local=True,
    ),
    
    # Google Gemini Models
    "gemini-2.0-flash-exp": ModelPricing(
        model_name="gemini-2.0-flash-exp",
        input_cost_per_million=0.075,   # $0.075 per 1M input tokens
        output_cost_per_million=0.30,   # $0.30 per 1M output tokens
        is_local=False,
    ),
    "gemini-1.5-pro": ModelPricing(
        model_name="gemini-1.5-pro",
        input_cost_per_million=1.25,    # $1.25 per 1M input tokens
        output_cost_per_million=5.00,   # $5.00 per 1M output tokens
        is_local=False,
    ),
    "gemini-2.5-pro": ModelPricing(
        model_name="gemini-2.5-pro",
        input_cost_per_million=1.25,    # Using same as 1.5-pro (update when pricing available)
        output_cost_per_million=10.00,  # Higher output cost for latest model
        is_local=False,
    ),
}

# Baseline model for savings calculation (most expensive option)
BASELINE_MODEL = "gemini-2.5-pro"


@dataclass
class CostEstimate:
    """
    Cost estimate for a request.
    
    Attributes:
        model: Model that was used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        estimated_cost: Cost in USD for this request
        baseline_cost: What it would have cost using baseline model
        savings: Difference between baseline and actual cost
        savings_percent: Percentage saved
    """
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    baseline_cost: float
    savings: float
    savings_percent: float


class CostCalculator:
    """
    Calculates costs and savings for LLM requests.
    """
    
    def __init__(self, pricing: Optional[dict[str, ModelPricing]] = None):
        """
        Initialize with pricing table.
        
        Args:
            pricing: Custom pricing table (defaults to MODEL_PRICING)
        """
        self.pricing = pricing or MODEL_PRICING
        self.baseline_model = BASELINE_MODEL
    
    def get_pricing(self, model: str) -> ModelPricing:
        """
        Get pricing for a model.
        
        Falls back to baseline pricing if model not found.
        """
        if model in self.pricing:
            return self.pricing[model]
        
        # Try partial match (e.g., "granite4:350m" matches "granite4")
        for key, pricing in self.pricing.items():
            if model.startswith(key.split(":")[0]):
                return pricing
        
        # Default to baseline pricing (conservative estimate)
        return self.pricing.get(self.baseline_model, ModelPricing(
            model_name=model,
            input_cost_per_million=1.25,
            output_cost_per_million=5.00,
        ))
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        pricing = self.get_pricing(model)
        
        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million
        
        return input_cost + output_cost
    
    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """
        Calculate full cost estimate with savings.
        
        Args:
            model: Model that was used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            CostEstimate with actual cost and savings vs baseline
        """
        actual_cost = self.calculate_cost(model, input_tokens, output_tokens)
        baseline_cost = self.calculate_cost(self.baseline_model, input_tokens, output_tokens)
        
        savings = baseline_cost - actual_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=actual_cost,
            baseline_cost=baseline_cost,
            savings=savings,
            savings_percent=savings_percent,
        )


class RequestLogger:
    """
    Logs requests to the database for tracking and analytics.
    """
    
    def __init__(self, calculator: Optional[CostCalculator] = None):
        """Initialize with optional custom calculator."""
        self.calculator = calculator or CostCalculator()
    
    async def log_request(
        self,
        session: AsyncSession,
        api_key_id: Optional[UUID],
        prompt: str,
        response_text: str,
        provider_response: ProviderResponse,
        tier: DifficultyTier,
        cache_hit: bool = False,
    ) -> RequestLog:
        """
        Log a completed request to the database.
        
        Args:
            session: Database session
            api_key_id: ID of the API key used
            prompt: Original prompt
            response_text: Generated response
            provider_response: Response from the provider
            tier: Difficulty tier used
            cache_hit: Whether response was from cache
            
        Returns:
            The created RequestLog record
        """
        # Calculate costs
        cost_estimate = self.calculator.estimate(
            model=provider_response.model,
            input_tokens=provider_response.prompt_tokens,
            output_tokens=provider_response.completion_tokens,
        )
        
        # Create log entry (using correct field names from RequestLog model)
        log = RequestLog(
            api_key_id=api_key_id,
            prompt_hash=self._hash_prompt(prompt),
            prompt_length=len(prompt),
            model_used=provider_response.model,
            difficulty_tag=tier.value,
            input_tokens=provider_response.prompt_tokens,
            output_tokens=provider_response.completion_tokens,
            estimated_cost=cost_estimate.estimated_cost,
            baseline_cost=cost_estimate.baseline_cost,
            cost_saved=cost_estimate.savings,
            latency_ms=provider_response.latency_ms,
            cache_hit=cache_hit,
        )
        
        session.add(log)
        return log
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for cache lookup."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()


# Singleton instances
_calculator: Optional[CostCalculator] = None
_logger: Optional[RequestLogger] = None


def get_cost_calculator() -> CostCalculator:
    """Get the singleton cost calculator."""
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator()
    return _calculator


def get_request_logger() -> RequestLogger:
    """Get the singleton request logger."""
    global _logger
    if _logger is None:
        _logger = RequestLogger()
    return _logger
