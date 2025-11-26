"""Global singletons and helpers."""

from __future__ import annotations

from typing import Optional

from ..llm_adapters.base import LLMClient
from ..llm_adapters.local_echo import LocalEchoLLMClient
from ..llm_adapters.ollama import OllamaLLMClient
from ..router.rule_based import RuleBasedMetricRouter
from .orchestrator import EvalOrchestrator
from .registry import MetricRegistry

_registry: Optional[MetricRegistry] = None
_router: Optional[RuleBasedMetricRouter] = None
_orchestrator: Optional[EvalOrchestrator] = None
_llm_client: Optional[LLMClient] = None


def get_registry() -> MetricRegistry:
    """Return the singleton registry, instantiating it if needed."""
    global _registry
    if _registry is None:
        _registry = MetricRegistry()
        from ..metrics import load_builtin_metrics

        load_builtin_metrics(_registry)
    return _registry


def get_router() -> RuleBasedMetricRouter:
    """Return the rule-based router singleton."""
    global _router
    if _router is None:
        _router = RuleBasedMetricRouter(get_registry())
    return _router


def get_orchestrator() -> EvalOrchestrator:
    """Return the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EvalOrchestrator(
            registry=get_registry(),
            router=get_router(),
            llm_client_resolver=get_llm_client,
        )
    return _orchestrator


def set_llm_client(client: Optional[LLMClient]) -> None:
    """Override the default LLM client."""
    global _llm_client
    _llm_client = client


def get_llm_client() -> LLMClient:
    """Return the configured LLM client or fall back to LocalEcho."""
    global _llm_client
    if _llm_client is None:
        try:
            _llm_client = OllamaLLMClient()
        except Exception:  # pragma: no cover - fallback path
            _llm_client = LocalEchoLLMClient()
    return _llm_client

