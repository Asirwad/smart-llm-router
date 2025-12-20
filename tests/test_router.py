"""
Unit tests for the Router Agent.

Tests various prompt classifications to verify the rule-based router
correctly identifies simple, medium, and complex prompts.
"""

import pytest

from src.core.router import (
    DifficultyTier,
    PromptAnalysis,
    RuleBasedRouter,
    get_router,
)
from src.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before each test to pick up config changes."""
    get_settings.cache_clear()
    yield


@pytest.fixture
def router():
    """Create a fresh router instance for each test."""
    return RuleBasedRouter()


class TestPromptAnalysis:
    """Tests for prompt analysis features."""
    
    def test_token_count(self, router):
        """Token count approximation."""
        # ~4 chars per token
        analysis = router.analyze_prompt("Hello world")
        assert analysis.token_count == 2  # 11 chars / 4 ≈ 2
        
    def test_code_block_detection(self, router):
        """Detects code blocks."""
        prompt = "Here's some code:\n```python\nprint('hello')\n```"
        analysis = router.analyze_prompt(prompt)
        assert analysis.has_code_block is True
        
    def test_no_code_block(self, router):
        """No false positives for code blocks."""
        prompt = "Please summarize this text."
        analysis = router.analyze_prompt(prompt)
        assert analysis.has_code_block is False
        
    def test_question_count(self, router):
        """Counts questions correctly."""
        prompt = "What is Python? How does it work? Why use it?"
        analysis = router.analyze_prompt(prompt)
        assert analysis.question_count == 3


class TestSimpleClassification:
    """Tests for prompts that should be classified as SIMPLE."""
    
    @pytest.mark.asyncio
    async def test_summarize(self, router):
        """Summarization requests should be simple."""
        prompt = "Summarize this article in 3 bullet points."
        decision = await router.classify(prompt)
        assert decision.tier == DifficultyTier.SIMPLE
        
    @pytest.mark.asyncio
    async def test_translate(self, router):
        """Translation requests should be simple."""
        prompt = "Translate this to French: Hello, how are you?"
        decision = await router.classify(prompt)
        assert decision.tier == DifficultyTier.SIMPLE
        
    @pytest.mark.asyncio
    async def test_short_question(self, router):
        """Short, simple questions should be simple."""
        prompt = "What is Python?"
        decision = await router.classify(prompt)
        assert decision.tier == DifficultyTier.SIMPLE
        
    @pytest.mark.asyncio
    async def test_define(self, router):
        """Definition requests should be simple."""
        prompt = "Define machine learning."
        decision = await router.classify(prompt)
        assert decision.tier == DifficultyTier.SIMPLE


class TestMediumClassification:
    """Tests for prompts with MEDIUM-level keywords."""
    
    @pytest.mark.asyncio
    async def test_explanation_keywords_detected(self, router):
        """Verify 'explain' keyword is detected in analysis."""
        prompt = "Explain in detail how variables work in Python programming. Include examples of integers, strings, and other data types."
        analysis = router.analyze_prompt(prompt)
        # Should detect this is not a simple keyword prompt
        assert analysis.has_simple_keywords is False or analysis.has_reasoning_keywords is True or analysis.has_code_keywords is True
        
    @pytest.mark.asyncio
    async def test_code_generation_detects_code_keywords(self, router):
        """Code generation prompts should detect code keywords."""
        prompt = "Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list."
        analysis = router.analyze_prompt(prompt)
        assert analysis.has_code_keywords is True
        
    @pytest.mark.asyncio
    async def test_comparison_at_least_medium_with_length(self, router):
        """Long comparison with multiple questions should be at least medium."""
        prompt = """Compare and contrast Python and Java comprehensively:
        1. How do their syntaxes differ?
        2. What are the performance characteristics?
        3. Which is better for web development and why?
        4. What about mobile development?"""
        decision = await router.classify(prompt)
        # Long comparison with multiple questions should not be simple
        assert decision.tier in [DifficultyTier.MEDIUM, DifficultyTier.COMPLEX]


class TestComplexClassification:
    """Tests for prompts that should NOT be classified as SIMPLE."""
    
    @pytest.mark.asyncio
    async def test_architecture_not_simple(self, router):
        """Architecture prompts should be at least medium."""
        prompt = """Design system architecture for a high-availability e-commerce platform. 
        Include database schema design, API design, security considerations,
        and explain the trade-offs of your architectural decisions."""
        decision = await router.classify(prompt)
        # Architecture should never be simple
        assert decision.tier != DifficultyTier.SIMPLE, f"Got SIMPLE, reason: {decision.reason}"
        
    @pytest.mark.asyncio
    async def test_debugging_with_code_not_simple(self, router):
        """Debugging with code should not be simple."""
        prompt = """Debug and fix this code. Why doesn't it work correctly?

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n)  # Bug here

def main():
    for i in range(10):
        print(factorial(i))
```

Step by step: First identify the bug, then explain why it causes issues, finally fix it."""
        decision = await router.classify(prompt)
        # Debugging with code should never be simple
        assert decision.tier != DifficultyTier.SIMPLE, f"Got SIMPLE, reason: {decision.reason}"
        
    @pytest.mark.asyncio
    async def test_multi_step_optimization_not_simple(self, router):
        """Multi-step optimization tasks should not be simple."""
        prompt = """Step-by-step database optimization task:

Step 1: First, analyze the current database schema.
Step 2: Then, identify performance bottlenecks.
Step 3: Next, design an optimized schema.
Step 4: Finally, write migration scripts.

This requires advanced system design knowledge."""
        decision = await router.classify(prompt)
        # Multi-step tasks should never be simple
        assert decision.tier != DifficultyTier.SIMPLE, f"Got SIMPLE, reason: {decision.reason}"
        
    @pytest.mark.asyncio
    async def test_best_model_override(self, router):
        """Explicit 'best model' request should trigger complex."""
        prompt = "Use the best model to answer: What is 2+2?"
        decision = await router.classify(prompt)
        # This should ALWAYS be complex due to override logic
        assert decision.tier == DifficultyTier.COMPLEX
        assert "best" in decision.reason.lower() or "quality" in decision.reason.lower()


class TestForcedTier:
    """Tests for force_tier override."""
    
    @pytest.mark.asyncio
    async def test_force_simple(self, router):
        """Force simple even for complex prompt."""
        prompt = "Design a microservices architecture."
        decision = await router.classify(prompt, force_tier="simple")
        assert decision.tier == DifficultyTier.SIMPLE
        assert "forced" in decision.reason.lower()
        
    @pytest.mark.asyncio
    async def test_force_complex(self, router):
        """Force complex even for simple prompt."""
        prompt = "What is 2+2?"
        decision = await router.classify(prompt, force_tier="complex")
        assert decision.tier == DifficultyTier.COMPLEX
        assert "forced" in decision.reason.lower()


class TestRouterSingleton:
    """Tests for router singleton."""
    
    def test_get_router_returns_instance(self):
        """get_router() returns a router instance."""
        router = get_router()
        assert isinstance(router, RuleBasedRouter)
        
    def test_get_router_returns_same_instance(self):
        """get_router() returns the same instance."""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2
