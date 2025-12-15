"""
Base protocol and factory for LLM clients.

Provides a unified interface (LlmClientProtocol) and factory function
(create_llm_client) for creating OpenAI and Gemini LLM clients.

Also provides function_to_tool_schema to automatically generate tool schemas
from Python function signatures and docstrings.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlalchemy.orm import Session

from src.agent.gpt_client import OpenAIOrchestratorClient


class LlmClientProtocol(Protocol):
    """Protocol defining the interface for LLM clients."""

    @property
    def config_path(self) -> Path | None:
        """Path to config file."""
        ...

    @property
    def config(self) -> Any:
        """Client configuration."""
        ...

    def function_to_tool_schema(
        self,
        func: Any,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """
        Generate tool schema from function signature in backend's native format.

        Args:
            func: The Python function to generate a schema for
            name: Optional function name override (defaults to func.__name__)
            description: Optional description override (extracted from docstring if not provided)

        Returns:
            Backend-specific tool format (dict for OpenAI, protobuf for Gemini)
        """
        ...

    async def chat_with_tools(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | type | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat with LLM supporting tool calling and structured outputs."""
        ...

    def _store_request_to_db(
        self,
        request_id: str,
        batch_id: str | None,
        response_id: str | None,
        context: dict[str, Any],
        request_data: dict[str, Any] | None = None,
        status: str = "pending",
        error_message: str | None = None,
    ) -> None:
        """Store request to database."""
        ...

    def _update_request_in_db(
        self,
        response_id: str | None = None,
        request_id: str | None = None,
        response_data: dict[str, Any] | None = None,
        usage: Any | None = None,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """Update request in database with response data."""
        ...


def create_llm_client(
    provider: Literal["openai", "gemini"],
    model: str,
    reasoning_effort: str = "medium",
    enable_web_search: bool = False,
    search_context_size: Literal["medium", "high"] = "medium",
    service_tier: Literal["flex", "default", "priority"] = "flex",
    timeout: int | None = None,
    config_path: Path | None = None,
    session: Session | None = None,
    temperature: float = 1,
    max_iterations: int = 10,
) -> LlmClientProtocol:
    """
    Factory function to create an LLM client based on provider.

    Args:
        provider: LLM provider ('openai' or 'gemini')
        model: Model name
        reasoning_effort: Reasoning effort level
        enable_web_search: Enable web search
        search_context_size: Search context size
        service_tier: Service tier
        timeout: Request timeout
        config_path: Path to config file
        session: Optional database session
        temperature: Temperature for Gemini models (0.0-2.0). Not used for GPT thinking models.
        max_iterations: Maximum number of iterations for function calling loop

    Returns:
        LLM client instance implementing LlmClientProtocol
    """
    if provider == "openai":
        return OpenAIOrchestratorClient(
            model=model,
            reasoning_effort=reasoning_effort,
            enable_web_search=enable_web_search,
            search_context_size=search_context_size,
            service_tier=service_tier,
            timeout=timeout,
            config_path=config_path,
            session=session,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'gemini'")
