"""
Router Agent for Smart Model Router.

This module classifies prompts and determines which model tier to use.
It's designed with an abstract interface so we can swap implementations:
- RuleBasedRouter: Current implementation using heuristics
- LLMRouter: Future implementation using lightweight LLM for classification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import get_settings


class DifficultyTier(str, Enum):
    """Difficulty tiers that map to different models."""
    SIMPLE = "simple"    # Local model (Granite 4.0 Nano)
    MEDIUM = "medium"    # Gemini Flash
    COMPLEX = "complex"  # Gemini Pro


@dataclass
class RoutingDecision:
    """
    Result of the routing decision.
    
    Attributes:
        tier: The difficulty tier (simple/medium/complex)
        model: The specific model to use
        reason: Human-readable explanation of why this tier was chosen
        confidence: How confident the router is (0.0 to 1.0)
    """
    tier: DifficultyTier
    model: str
    reason: str
    confidence: float = 1.0


@dataclass
class PromptAnalysis:
    """
    Analysis of a prompt's characteristics.
    
    Used by the router to make decisions.
    """
    token_count: int
    word_count: int
    has_code_block: bool
    has_code_keywords: bool
    has_reasoning_keywords: bool
    has_simple_keywords: bool
    question_count: int
    instruction_complexity: int  # 1-5 scale


class BaseRouter(ABC):
    """
    Abstract base class for routers.
    
    Allows swapping between rule-based and LLM-based routing.
    """
    
    @abstractmethod
    async def classify(self, prompt: str, force_tier: Optional[str] = None) -> RoutingDecision:
        """
        Classify a prompt and return routing decision.
        
        Args:
            prompt: The user's prompt text
            force_tier: Optional override to force a specific tier
            
        Returns:
            RoutingDecision with tier, model, and reasoning
        """
        pass
    
    @abstractmethod
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Analyze prompt characteristics.
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            PromptAnalysis with extracted features
        """
        pass


class RuleBasedRouter(BaseRouter):
    """
    Rule-based router using heuristics.
    
    Classifies prompts based on:
    - Token/word count
    - Presence of code
    - Keywords indicating complexity
    - Question structure
    """
    
    # Keywords indicating simple tasks
    SIMPLE_KEYWORDS = {
        "summarize", "summary", "summarise", "tldr", "tl;dr",
        "translate", "translation",
        "rewrite", "rephrase", "paraphrase",
        "extract", "list", "bullet points",
        "define", "definition", "what is", "what are",
        "format", "convert", "fix grammar", "proofread",
        "yes or no", "true or false",
    }
    
    # Keywords indicating medium complexity
    MEDIUM_KEYWORDS = {
        "explain", "describe", "compare", "contrast",
        "analyze", "analyse", "evaluate",
        "write a", "create a", "generate",
        "how to", "how do", "how does", "how can",
        "why", "what if",
        "example", "examples",
        "code", "function", "script", "program",
        "json", "xml", "yaml", "csv",
    }
    
    # Keywords indicating complex tasks
    COMPLEX_KEYWORDS = {
        "architect", "architecture", "design system",
        "debug", "fix this", "why isn't", "why doesn't",
        "optimize", "optimise", "performance",
        "refactor", "restructure", "redesign",
        "multiple steps", "step by step", "step-by-step",
        "pros and cons", "trade-offs", "tradeoffs",
        "implement", "build", "develop",
        "algorithm", "data structure",
        "best model", "highest quality", "most accurate",
        "complex", "advanced", "sophisticated",
        "security", "authentication", "authorization",
        "database schema", "api design", "system design",
    }
    
    # Code-related patterns
    CODE_PATTERNS = {
        "```", "def ", "function ", "class ",
        "import ", "from ", "const ", "let ", "var ",
        "public ", "private ", "async ", "await ",
        "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
        "CREATE TABLE", "ALTER TABLE",
    }
    
    def __init__(self):
        """Initialize with settings."""
        self.settings = get_settings()
        
        # Model mapping
        self.model_map = {
            DifficultyTier.SIMPLE: self.settings.ollama_model,
            DifficultyTier.MEDIUM: self.settings.gemini_flash_model,
            DifficultyTier.COMPLEX: self.settings.gemini_pro_model,
        }
    
    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count.
        
        Simple heuristic: ~4 characters per token for English.
        For accurate counts, we'd use tiktoken, but this is faster.
        """
        return len(text) // 4
    
    def _has_keywords(self, text_lower: str, keywords: set) -> bool:
        """Check if text contains any of the keywords."""
        return any(kw in text_lower for kw in keywords)
    
    def _count_questions(self, text: str) -> int:
        """Count the number of questions in the text."""
        return text.count("?")
    
    def _has_code_block(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return "```" in text or text.count("    ") > 3  # Indented code
    
    def _has_code_patterns(self, text: str) -> bool:
        """Check for code-like patterns."""
        return any(pattern in text for pattern in self.CODE_PATTERNS)
    
    def _estimate_instruction_complexity(self, text: str) -> int:
        """
        Estimate instruction complexity on a 1-5 scale.
        
        Based on:
        - Number of distinct requests/tasks
        - Presence of conditional logic (if, when, unless)
        - Multi-step requirements
        """
        text_lower = text.lower()
        score = 1
        
        # Multiple requests?
        if text.count(". ") > 3 or text.count("\n") > 5:
            score += 1
        
        # Conditional logic?
        conditionals = ["if ", "when ", "unless ", "only if", "except"]
        if any(c in text_lower for c in conditionals):
            score += 1
        
        # Multi-step?
        step_indicators = ["first", "then", "next", "finally", "step 1", "1.", "2.", "3."]
        if sum(1 for s in step_indicators if s in text_lower) >= 2:
            score += 1
        
        # Very long prompt?
        if len(text) > 2000:
            score += 1
        
        return min(score, 5)
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Analyze the prompt and extract features."""
        text_lower = prompt.lower()
        
        return PromptAnalysis(
            token_count=self._count_tokens(prompt),
            word_count=len(prompt.split()),
            has_code_block=self._has_code_block(prompt),
            has_code_keywords=self._has_code_patterns(prompt),
            has_reasoning_keywords=self._has_keywords(text_lower, self.COMPLEX_KEYWORDS),
            has_simple_keywords=self._has_keywords(text_lower, self.SIMPLE_KEYWORDS),
            question_count=self._count_questions(prompt),
            instruction_complexity=self._estimate_instruction_complexity(prompt),
        )
    
    async def classify(self, prompt: str, force_tier: Optional[str] = None) -> RoutingDecision:
        """
        Classify the prompt using rule-based heuristics.
        
        Decision flow:
        1. Check for forced tier override
        2. Check for "best model" override keywords
        3. Analyze prompt features
        4. Apply scoring rules
        5. Return decision with explanation
        """
        settings = self.settings
        
        # 1. Handle forced tier
        if force_tier:
            tier = DifficultyTier(force_tier)
            return RoutingDecision(
                tier=tier,
                model=self.model_map[tier],
                reason=f"Forced to {force_tier} tier by request",
                confidence=1.0,
            )
        
        # 2. Check for "best model" override
        text_lower = prompt.lower()
        best_model_triggers = ["best model", "highest quality", "most accurate", "gpt-4", "most capable"]
        if any(trigger in text_lower for trigger in best_model_triggers):
            return RoutingDecision(
                tier=DifficultyTier.COMPLEX,
                model=self.model_map[DifficultyTier.COMPLEX],
                reason="User requested best/highest quality model",
                confidence=0.95,
            )
        
        # 3. Analyze prompt
        analysis = self.analyze_prompt(prompt)
        
        # 4. Scoring logic
        score = 0  # Start neutral, negative = simple, positive = complex
        reasons = []
        
        # Token count thresholds
        if analysis.token_count < settings.simple_token_threshold:
            score -= 2
            reasons.append(f"Short prompt ({analysis.token_count} tokens)")
        elif analysis.token_count > settings.complex_token_threshold:
            score += 2
            reasons.append(f"Long prompt ({analysis.token_count} tokens)")
        
        # Code detection
        if analysis.has_code_block:
            score += 2
            reasons.append("Contains code blocks")
        elif analysis.has_code_keywords:
            score += 1
            reasons.append("Contains code patterns")
        
        # Keyword detection
        if analysis.has_simple_keywords and not analysis.has_reasoning_keywords:
            score -= 2
            reasons.append("Simple task keywords detected")
        
        if analysis.has_reasoning_keywords:
            score += 2
            reasons.append("Complex reasoning keywords detected")
        
        # Instruction complexity
        if analysis.instruction_complexity >= 4:
            score += 2
            reasons.append(f"High instruction complexity ({analysis.instruction_complexity}/5)")
        elif analysis.instruction_complexity <= 2:
            score -= 1
            reasons.append(f"Low instruction complexity ({analysis.instruction_complexity}/5)")
        
        # Multiple questions = more complex
        if analysis.question_count > 3:
            score += 1
            reasons.append(f"Multiple questions ({analysis.question_count})")
        
        # 5. Map score to tier
        if score <= -2:
            tier = DifficultyTier.SIMPLE
        elif score >= 3:
            tier = DifficultyTier.COMPLEX
        else:
            tier = DifficultyTier.MEDIUM
        
        # Calculate confidence based on how decisive the score is
        confidence = min(0.9, 0.5 + abs(score) * 0.1)
        
        return RoutingDecision(
            tier=tier,
            model=self.model_map[tier],
            reason="; ".join(reasons) if reasons else "Default classification",
            confidence=confidence,
        )


# Default router instance
_router: Optional[BaseRouter] = None
_rule_router: Optional[RuleBasedRouter] = None


class LLMRouter(BaseRouter):
    """
    LLM-based router using Granite for classification.
    
    Uses the local Granite model to analyze prompt complexity.
    Falls back to rule-based routing if LLM fails.
    """
    
    # Classification prompt template
    CLASSIFICATION_PROMPT = """You are a prompt classifier. Analyze the following user prompt and classify its complexity.

User Prompt:
{prompt}

Classify this prompt into ONE of these categories:
- SIMPLE: Basic tasks like summarization, translation, simple Q&A, formatting
- MEDIUM: Explanations, code generation, comparisons, moderate analysis
- COMPLEX: Architecture design, debugging, multi-step reasoning, advanced analysis

Respond with ONLY the category name (SIMPLE, MEDIUM, or COMPLEX) and a brief reason.
Format: CATEGORY|reason

Example: SIMPLE|This is a basic translation request"""

    def __init__(self):
        """Initialize with fallback router."""
        self.settings = get_settings()
        self._fallback_router = RuleBasedRouter()
        self._client = None
        
        # Model mapping
        self.model_map = {
            DifficultyTier.SIMPLE: self.settings.ollama_model,
            DifficultyTier.MEDIUM: self.settings.gemini_flash_model,
            DifficultyTier.COMPLEX: self.settings.gemini_pro_model,
        }
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30)
        return self._client
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Delegate to rule-based router for analysis."""
        return self._fallback_router.analyze_prompt(prompt)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Granite for classification."""
        client = await self._get_client()
        
        classification_prompt = self.CLASSIFICATION_PROMPT.format(prompt=prompt[:1000])  # Limit prompt size
        
        response = await client.post(
            f"{self.settings.ollama_base_url}/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": classification_prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    
    def _parse_classification(self, llm_response: str) -> tuple[DifficultyTier, str]:
        """Parse LLM response into tier and reason."""
        response = llm_response.strip().upper()
        
        # Try to parse CATEGORY|reason format
        if "|" in response:
            parts = response.split("|", 1)
            category = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else "LLM classification"
        else:
            # Just extract the category from the response
            category = response
            reason = "LLM classification"
        
        # Map to tier
        if "SIMPLE" in category:
            return DifficultyTier.SIMPLE, reason
        elif "COMPLEX" in category:
            return DifficultyTier.COMPLEX, reason
        else:
            return DifficultyTier.MEDIUM, reason
    
    async def classify(self, prompt: str, force_tier: Optional[str] = None) -> RoutingDecision:
        """
        Classify using LLM with fallback to rules.
        """
        # Handle forced tier
        if force_tier:
            tier = DifficultyTier(force_tier)
            return RoutingDecision(
                tier=tier,
                model=self.model_map[tier],
                reason=f"Forced to {force_tier} tier by request",
                confidence=1.0,
            )
        
        try:
            # Call LLM for classification
            llm_response = await self._call_llm(prompt)
            tier, reason = self._parse_classification(llm_response)
            
            return RoutingDecision(
                tier=tier,
                model=self.model_map[tier],
                reason=f"LLM: {reason}",
                confidence=0.85,
            )
        except Exception as e:
            # Fallback to rule-based on any error
            print(f"⚠️ LLM classification failed, using rules: {e}")
            return await self._fallback_router.classify(prompt, force_tier)


def get_router() -> BaseRouter:
    """
    Get the singleton router instance.
    
    Uses LLMRouter if enabled in settings, otherwise RuleBasedRouter.
    """
    global _router, _rule_router
    
    settings = get_settings()
    
    if settings.use_llm_router:
        if _router is None or not isinstance(_router, LLMRouter):
            _router = LLMRouter()
        return _router
    else:
        if _rule_router is None:
            _rule_router = RuleBasedRouter()
        return _rule_router

