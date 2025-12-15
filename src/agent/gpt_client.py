"""
OpenAI GPT LLM client implementation for vendor processing.

Provides async OpenAI client with web search support, batch processing (limited for prebid extraction),
and request/response tracking in the database.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.responses import ResponseUsage

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.agent.data_models import LinkVendorDecision, UrlCitation
from src.helper.db_operations import (
    get_llm_request_by_response_id,
    get_llm_requests_by_batch_id,
    store_llm_request,
    transactional_session,
)
from src.helper.db_schema import LlmRequest

logger = logging.getLogger(__name__)


@dataclass
class LlmClientConfig:
    """Configuration for the LLM client."""

    model: str
    reasoning_effort: str = "medium"
    enable_web_search: bool = False
    search_context_size: Literal["medium", "high"] = "medium"
    service_tier: Literal["flex", "default", "priority"] = "flex"
    timeout: int | None = None
    background: bool = False


@dataclass
class BatchResult:
    """Result from batch processing."""

    response_id: str | None
    context: dict[str, Any]
    request_data: dict[str, Any] | None = None
    batch_id: str | None = None
    request_id: str | None = None


@dataclass
class CostSummary:
    """Summary of API costs."""

    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    request_count: int


class OpenAIOrchestratorClient:
    """Async OpenAI client for vendor linking decisions with web search and batch processing."""

    def __init__(
        self,
        model: str,
        reasoning_effort: str = "medium",
        enable_web_search: bool = False,
        search_context_size: Literal["medium", "high"] = "medium",
        service_tier: Literal["flex", "default", "priority"] = "flex",
        timeout: int | None = None,
        config_path: Path | None = None,
        session: Session | None = None,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.config = LlmClientConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            enable_web_search=enable_web_search,
            search_context_size=search_context_size,
            service_tier=service_tier,
            timeout=timeout,
        )
        self.config_path = config_path
        self.session = session  # Optional session to reuse (avoids SQLite locks)
        self._total_cost: float = 0.0
        self._total_requests: int = 0

    def _build_reasoning_config(self) -> dict[str, Any] | None:
        """Build reasoning configuration based on model and settings."""
        reasoning: dict[str, Any] = {}
        if self.config.reasoning_effort:
            reasoning["effort"] = self.config.reasoning_effort
        # Always include summary for better reasoning visibility
        reasoning["summary"] = "auto"
        return reasoning if reasoning else None

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
        # Use provided session if available, otherwise create a new one
        if self.session:
            try:
                store_llm_request(
                    self.session,
                    {
                        "id": request_id,
                        "batch_id": batch_id,
                        "response_id": response_id,
                        "model": self.config.model,
                        "service_tier": self.config.service_tier,
                        "status": status,
                        "context": context,
                        "request_data": request_data,
                        "error_message": error_message,
                    },
                )
                # Flush immediately so it's available, but don't commit (let caller handle commit)
                self.session.flush()
            except OperationalError as e:
                # Database locked - non-critical, just log warning
                logger.warning(f"Could not store request to DB (database locked): {e}")
            except Exception as e:
                logger.error(f"Error storing request to DB: {e}", exc_info=True)
            return

        # Fallback: create new session if no session provided
        if not self.config_path:
            logger.warning("config_path not set, skipping DB storage")
            return

        try:
            # Convert directory path to file path if needed
            config_file_path = self.config_path
            if config_file_path.is_dir():
                config_file_path = config_file_path / "config.yaml"
            with transactional_session(config_file_path) as session:
                store_llm_request(
                    session,
                    {
                        "id": request_id,
                        "batch_id": batch_id,
                        "response_id": response_id,
                        "model": self.config.model,
                        "service_tier": self.config.service_tier,
                        "status": status,
                        "context": context,
                        "request_data": request_data,
                        "error_message": error_message,
                    },
                )
        except OperationalError as e:
            # Database locked - non-critical, just log warning
            logger.warning(f"Could not store request to DB (database locked): {e}")
        except Exception as e:
            logger.error(f"Error storing request to DB: {e}", exc_info=True)

    def _build_update_data(
        self,
        request: LlmRequest,
        response_id: str | None,
        response_data: dict[str, Any] | None,
        usage: ResponseUsage | None,
        status: str,
        error_message: str | None,
    ) -> dict[str, Any]:
        """Build update data dictionary for LLM request."""
        update_data: dict[str, Any] = {
            "id": request.id,
            "status": status,
            "response_data": response_data,
            "error_message": error_message,
        }

        # Update response_id if provided and not already set
        if response_id and not request.response_id:
            update_data["response_id"] = response_id

        if usage:
            cost = self.calculate_cost(
                usage, self.config.model, self.config.service_tier
            )
            input_tokens = usage.input_tokens - (
                usage.input_tokens_details.cached_tokens
                if usage.input_tokens_details
                else 0
            )
            output_tokens = usage.output_tokens
            reasoning_tokens = (
                usage.output_tokens_details.reasoning_tokens
                if usage.output_tokens_details
                else 0
            )

            update_data.update(
                {
                    "cost": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                }
            )

        if status == "completed":
            update_data["completed_at"] = datetime.now(UTC)

        return update_data

    def _update_request_in_db(
        self,
        response_id: str | None = None,
        request_id: str | None = None,
        response_data: dict[str, Any] | None = None,
        usage: ResponseUsage | None = None,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """Update request in database with response data."""
        if not response_id and not request_id:
            logger.warning("Either response_id or request_id must be provided")
            return

        # Use provided session if available, otherwise create a new one
        if self.session:
            try:
                # Prioritize request_id lookup since requests are stored with request_id first
                # Only use response_id if request_id is not available
                if request_id:
                    request = self.session.get(LlmRequest, request_id)
                elif response_id:
                    request = get_llm_request_by_response_id(self.session, response_id)
                else:
                    logger.warning("Neither request_id nor response_id provided")
                    return

                if not request:
                    identifier = response_id or request_id
                    logger.warning(f"Request not found for {'response_id' if response_id else 'request_id'}: {identifier}")
                    return

                update_data = self._build_update_data(
                    request, response_id, response_data, usage, status, error_message
                )

                store_llm_request(self.session, update_data)
                # Flush immediately so it's available, but don't commit (let caller handle commit)
                self.session.flush()
            except OperationalError as e:
                # Database locked - non-critical, just log warning
                logger.warning(f"Could not update request in DB (database locked): {e}")
            except Exception as e:
                logger.error(f"Error updating request in DB: {e}", exc_info=True)
            return

        # Fallback: create new session if no session provided
        if not self.config_path:
            logger.warning("config_path not set, skipping DB update")
            return

        try:
            # Convert directory path to file path if needed
            config_file_path = self.config_path
            if config_file_path.is_dir():
                config_file_path = config_file_path / "config.yaml"
            with transactional_session(config_file_path) as session:
                # Prioritize request_id lookup since requests are stored with request_id first
                # Only use response_id if request_id is not available
                if request_id:
                    request = session.get(LlmRequest, request_id)
                elif response_id:
                    request = get_llm_request_by_response_id(session, response_id)
                else:
                    logger.warning("Neither request_id nor response_id provided")
                    return

                if not request:
                    identifier = response_id or request_id
                    logger.warning(f"Request not found for {'response_id' if response_id else 'request_id'}: {identifier}")
                    return

                update_data = self._build_update_data(
                    request, response_id, response_data, usage, status, error_message
                )

                store_llm_request(session, update_data)
        except OperationalError as e:
            # Database locked - non-critical, just log warning
            logger.warning(f"Could not update request in DB (database locked): {e}")
        except Exception as e:
            logger.error(f"Error updating request in DB: {e}", exc_info=True)

    async def get_background_result(
        self,
        response_id: str,
    ) -> tuple[
        LinkVendorDecision | None,
        ResponseUsage | None,
        Literal["completed", "pending", "failed"],
    ]:
        """
        Retrieve the result of a background request.

        Args:
            response_id: The response ID from a background request

        Returns:
            Tuple of (LinkVendorDecision or None, usage or None, status)
        """
        try:
            response = await self.client.responses.retrieve(response_id)

            if response.status != "completed":
                logger.info(
                    f"Background response {response_id} is still {response.status}"
                )
                # Update DB status
                self._update_request_in_db(response_id, status="pending")
                return None, None, "pending"

            output = response.output_text
            if not output:
                logger.warning(
                    f"No output text in completed background response {response_id}"
                )
                self._update_request_in_db(
                    response_id, status="failed", error_message="No output text"
                )
                return None, response.usage, "failed"

            # Update DB with response data
            response_dict = {
                "id": response.id,
                "status": getattr(response, "status", None),
                "output_text": response.output_text,
                "model": getattr(response, "model", None),
                "created_at": getattr(response, "created_at", None),
            }
            self._update_request_in_db(
                response_id=response_id,
                response_data=response_dict,
                usage=response.usage,
                status="completed",
            )

            # Track costs
            if response.usage:
                cost = self.calculate_cost(
                    response.usage, self.config.model, self.config.service_tier
                )
                self._total_cost += cost
                self._total_requests += 1

            decision = LinkVendorDecision.model_validate_json(output)
            return decision, response.usage, "completed"

        except RateLimitError as e:
            logger.error(
                f"Rate limit error retrieving background result {response_id}: {e}"
            )
            self._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"
        except APITimeoutError as e:
            logger.error(
                f"API timeout error retrieving background result {response_id}: {e}", exc_info=True
            )
            self._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"
        except APIConnectionError as e:
            logger.error(
                f"API connection error retrieving background result {response_id}: {e}", exc_info=True
            )
            self._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"
        except APIStatusError as e:
            logger.error(
                f"API status error retrieving background result {response_id}: {e.status_code} - {e}", exc_info=True
            )
            self._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"
        except OpenAIError as e:
            logger.error(
                f"OpenAI error retrieving background result {response_id}: {e}", exc_info=True
            )
            self._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"

    def get_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all requests for a batch ID from the database.

        Args:
            batch_id: The batch ID

        Returns:
            List of request dictionaries with all stored data
        """
        if not self.config_path:
            logger.warning("config_path not set, cannot retrieve batch results")
            return []

        try:
            with transactional_session(self.config_path) as session:
                requests = get_llm_requests_by_batch_id(session, batch_id)
                return [
                    {
                        "id": req.id,
                        "batch_id": req.batch_id,
                        "response_id": req.response_id,
                        "model": req.model,
                        "service_tier": req.service_tier,
                        "status": req.status,
                        "context": req.context,
                        "request_data": req.request_data,
                        "response_data": req.response_data,
                        "cost": req.cost,
                        "input_tokens": req.input_tokens,
                        "output_tokens": req.output_tokens,
                        "reasoning_tokens": req.reasoning_tokens,
                        "error_message": req.error_message,
                        "created_at": req.created_at.isoformat()
                        if req.created_at
                        else None,
                        "updated_at": req.updated_at.isoformat()
                        if req.updated_at
                        else None,
                        "completed_at": req.completed_at.isoformat()
                        if req.completed_at
                        else None,
                    }
                    for req in requests
                ]
        except Exception as e:
            logger.error(f"Error retrieving batch results: {e}", exc_info=True)
            return []

    def get_batch_cost_summary(self, batch_id: str) -> CostSummary:
        """
        Get cost summary for a specific batch.

        Args:
            batch_id: The batch ID

        Returns:
            CostSummary with aggregated costs for the batch
        """
        if not self.config_path:
            logger.warning("config_path not set, cannot retrieve batch cost summary")
            return CostSummary(0.0, 0, 0, 0, 0)

        try:
            with transactional_session(self.config_path) as session:
                requests = get_llm_requests_by_batch_id(session, batch_id)
                total_cost = sum(req.cost or 0.0 for req in requests)
                total_input_tokens = sum(req.input_tokens or 0 for req in requests)
                total_output_tokens = sum(req.output_tokens or 0 for req in requests)
                total_reasoning_tokens = sum(
                    req.reasoning_tokens or 0 for req in requests
                )
                request_count = len(requests)

                return CostSummary(
                    total_cost=total_cost,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    total_reasoning_tokens=total_reasoning_tokens,
                    request_count=request_count,
                )
        except Exception as e:
            logger.error(f"Error retrieving batch cost summary: {e}", exc_info=True)
            return CostSummary(0.0, 0, 0, 0, 0)

    def calculate_cost(
        self,
        usage: ResponseUsage,
        model: str | None = None,
        service_tier: str | None = None,
    ) -> float:
        """
        Calculate the cost of API usage based on current pricing per 1M tokens.

        Args:
            usage: ResponseUsage object containing token counts
            model: Model name (overrides config if provided)
            service_tier: Service tier ('flex', 'normal', or 'priority'). Overrides config if provided.

        Returns:
            Total cost in USD
        """
        model_name = model or self.config.model
        tier = service_tier or self.config.service_tier

        # Base pricing per 1M tokens (flex pricing)
        base_pricing = {
            "gpt-5": {
                "input": 0.625 / 1_000_000,
                "input_cached": 0.0625 / 1_000_000,
                "output": 5.00 / 1_000_000,
            },
            "gpt-5-mini": {
                "input": 0.125 / 1_000_000,
                "input_cached": 0.0125 / 1_000_000,
                "output": 1.00 / 1_000_000,
            },
            "gpt-5-nano": {
                "input": 0.025 / 1_000_000,
                "input_cached": 0.0025 / 1_000_000,
                "output": 0.20 / 1_000_000,
            },
            "o3": {
                "input": 1.00 / 1_000_000,
                "input_cached": 0.25 / 1_000_000,
                "output": 4.00 / 1_000_000,
            },
            "o4-mini": {
                "input": 0.55 / 1_000_000,
                "input_cached": 0.138 / 1_000_000,
                "output": 2.20 / 1_000_000,
            },
        }

        # Get service tier multiplier
        if tier.lower() == "flex":
            multiplier = 0.5
        elif tier.lower() == "default":
            multiplier = 1.0
        elif tier.lower() == "priority":
            multiplier = 2.0
        else:
            raise ValueError(f"Unknown service tier: {tier}")

        # Normalize model name
        model_key = model_name.lower()
        if "gpt-5-mini" in model_key:
            model_key = "gpt-5-mini"
        elif "gpt-5-nano" in model_key:
            model_key = "gpt-5-nano"
        elif "gpt-5" in model_key and model_key not in ["gpt-5-mini", "gpt-5-nano"]:
            model_key = "gpt-5"
        elif "o3" in model_key:
            model_key = "o3"
        elif "o4-mini" in model_key:
            model_key = "o4-mini"

        if model_key not in base_pricing:
            raise ValueError(
                f"Unknown model: {model_name}. Supported models: {list(base_pricing.keys())}. Service tier: {tier}"
            )

        costs = base_pricing[model_key]
        pricing = {
            "input": costs["input"] * multiplier,
            "input_cached": costs["input_cached"] * multiplier,
            "output": costs["output"] * multiplier,
        }

        # Calculate token usage
        input_tokens = usage.input_tokens - (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details
            else 0
        )
        input_cached_tokens = (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details
            else 0
        )
        output_tokens = usage.output_tokens
        reasoning_tokens = (
            usage.output_tokens_details.reasoning_tokens
            if usage.output_tokens_details
            else 0
        )

        # Calculate total cost
        total_cost = (
            (input_tokens * pricing["input"])
            + (input_cached_tokens * pricing["input_cached"])
            + (output_tokens * pricing["output"])
            + (
                reasoning_tokens * pricing["output"]
            )  # Reasoning tokens use output pricing
        )

        return round(total_cost, 6)

    def get_cost_summary(self) -> CostSummary:
        """
        Get summary of accumulated costs.

        Returns:
            CostSummary with total costs and token counts
        """
        return CostSummary(
            total_cost=self._total_cost,
            total_input_tokens=0,  # Not tracked individually, only cost
            total_output_tokens=0,  # Not tracked individually, only cost
            total_reasoning_tokens=0,  # Not tracked individually, only cost
            request_count=self._total_requests,
        )

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters."""
        self._total_cost = 0.0
        self._total_requests = 0

    def extract_evidence_from_response(
        self,
        response: Any,
        output_text: str,
        evidence_refs: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Extract evidence URLs from GPT annotations and match them to evidence refs.

        Args:
            response: The OpenAI response object
            output_text: The text output from the response
            evidence_refs: List of EvidenceRef objects from the parsed result

        Returns:
            List of evidence dictionaries with matched URLs
        """
        if not hasattr(response, "output") or not response.output:
            return []

        # Extract annotations from response
        annotations = []
        for output in response.output:
            if not hasattr(output, "content") or output.content is None:
                continue
            for content in output.content:
                if hasattr(content, "annotations"):
                    annotations.extend(content.annotations)

        # Process URL citations from annotations
        url_citations: list[UrlCitation] = []
        for annotation in annotations:
            if hasattr(annotation, "type") and annotation.type == "url_citation":
                url = annotation.url
                if url:
                    url = url.replace("?utm_source=chatgpt.com", "")
                    url = url.replace("?utm_source=openai", "")
                title = getattr(annotation, "title", "")
                start = getattr(annotation, "start_index", 0)
                end = getattr(annotation, "end_index", 0)
                reference_text = output_text[start:end] if output_text and start < len(output_text) and end <= len(output_text) else ""

                # Extract GPT-specific annotation ID if available
                annotation_id = getattr(annotation, "id", None)
                if annotation_id:
                    annotation_id = str(annotation_id)

                url_citations.append(UrlCitation(
                    url=url,
                    title=title,
                    start_index=start,
                    end_index=end,
                    reference_text=reference_text,
                    confidence=None,  # GPT annotations don't provide confidence
                    gpt_annotation_id=annotation_id,
                    gemini_chunk_index=None,
                    gemini_grounding_support=None,
                ))

        # Match citations to evidence refs
        evidence_updates = []
        for citation in url_citations:
            # Try to match citation to evidence by reference_text similarity
            evidence_matched = False
            for evidence_ref in evidence_refs:
                if hasattr(evidence_ref, "source_text") and evidence_ref.source_text:
                    # Simple similarity check - check if reference_text contains source_text or vice versa
                    if citation.reference_text and evidence_ref.source_text:
                        ref_lower = citation.reference_text.lower()
                        source_lower = evidence_ref.source_text.lower()
                        if ref_lower in source_lower or source_lower in ref_lower:
                            evidence_updates.append({
                                "ref_id": evidence_ref.ref_id,
                                "url": citation.url,
                                "title": citation.title,
                                "excerpt": citation.reference_text,
                                "confidence": evidence_ref.confidence,  # Use EvidenceRef confidence when matched
                            })
                            evidence_matched = True
                            break

            # If no match by text, try matching by title
            if not evidence_matched and citation.title:
                for evidence_ref in evidence_refs:
                    if hasattr(evidence_ref, "source_text") and evidence_ref.source_text:
                        source_lower = evidence_ref.source_text.lower()
                        title_lower = citation.title.lower()
                        if title_lower in source_lower or source_lower in title_lower:
                            evidence_updates.append({
                                "ref_id": evidence_ref.ref_id,
                                "url": citation.url,
                                "title": citation.title,
                                "excerpt": citation.reference_text,
                                "confidence": evidence_ref.confidence,  # Use EvidenceRef confidence when matched
                            })
                            evidence_matched = True
                            break

            # If still no match, create new evidence entry with citation's confidence (None for GPT)
            if not evidence_matched:
                logger.debug(f"No matching evidence found for URL citation: {citation.url}")
                evidence_updates.append({
                    "ref_id": None,  # New evidence, not linked to existing ref
                    "url": citation.url,
                    "title": citation.title,
                    "excerpt": citation.reference_text,
                    "confidence": citation.confidence,  # Use citation confidence when not matched (None for GPT)
                })

        return evidence_updates


    async def chat_with_tools(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | type | None = None,  # Can be dict or Pydantic model class
        context: dict[str, Any] | None = None,  # Optional context for request storage (e.g., cluster_id, iteration)
    ) -> dict[str, Any]:
        """
        Chat with LLM supporting tool calling and structured outputs.

        Args:
            messages: String (single message) or list of message dicts with 'role' and 'content' keys.
                     For responses API, list format with system/user roles is preferred.
            tools: List of tool definitions (OpenAI tool format)
            tool_choice: Tool choice setting ('auto', 'none', or dict with tool name)
            response_format: Optional response format (e.g., JSON schema for structured output)

        Returns:
            Dict containing:
                - response: The response object
                - output_text: The text output (if any)
                - tool_calls: List of tool calls (if any)
                - usage: Usage information
        """
        """
        Chat with LLM supporting tool calling and structured outputs.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: List of tool definitions (OpenAI tool format)
            tool_choice: Tool choice setting ('auto', 'none', or dict with tool name)
            response_format: Optional response format (e.g., JSON schema for structured output)

        Returns:
            Dict containing:
                - response: The response object
                - output_text: The text output (if any)
                - tool_calls: List of tool calls (if any)
                - usage: Usage information
        """
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": messages,
            "service_tier": self.config.service_tier,
        }

        # Add tools if provided
        # Responses API requires function tools to have 'name' at top level (not nested in "function")
        if tools:
            # Convert tools from chat.completions format to responses API format
            converted_tools = []
            for tool in tools:
                if isinstance(tool, dict) and tool["type"] == "function":
                    # Extract function definition
                    function_def = tool["function"]
                    if not isinstance(function_def, dict) or "name" not in function_def:
                        logger.error(f"Invalid function tool definition: {tool}")
                        raise ValueError(f"Function tool missing 'name' field: {tool}")

                    # Responses API expects name, description, parameters at top level
                    converted_tool: dict[str, Any] = {
                        "type": "function",
                        "name": function_def["name"],  # Required: name at top level
                        "parameters": function_def["parameters"] if "parameters" in function_def else {},
                    }
                    if "description" in function_def:
                        converted_tool["description"] = function_def["description"]
                    converted_tools.append(converted_tool)
                else:
                    # Non-function tools (like web_search) pass through as-is
                    converted_tools.append(tool)
            if converted_tools:
                create_kwargs["tools"] = converted_tools  # pyright: ignore[reportArgumentType]

        # Add tool_choice if provided
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice  # pyright: ignore[reportArgumentType]

        # Add structured output format if provided
        if response_format:
            if isinstance(response_format, type):
                # Pydantic model - convert to JSON schema
                schema = response_format.model_json_schema()
                model_name = getattr(response_format, "__name__", "response")

                # Recursively add additionalProperties: false to all object definitions to fix:
                # openai.BadRequestError: Error code: 400 - {'error':
                # {'message': "Invalid schema for response_format 'clusteranalysisresult':
                #  In context=(), 'additionalProperties' is required to be supplied and to be false.",
                #  'type': 'invalid_request_error', 'param': 'text.format.schema', 'code': 'invalid_json_schema'}}
                def add_additional_properties_false(obj: dict[str, Any]) -> dict[str, Any]:
                    """Recursively add additionalProperties: false to all object definitions."""
                    if isinstance(obj, dict):
                        result = {}
                        for key, value in obj.items():
                            if key == "properties" and isinstance(value, dict):
                                # This is an object definition - add additionalProperties
                                result[key] = {k: add_additional_properties_false(v) for k, v in value.items()}
                                result["additionalProperties"] = False
                            elif key == "items" and isinstance(value, dict):
                                # Array items - recurse
                                result[key] = add_additional_properties_false(value)
                            elif key in {"anyOf", "allOf", "oneOf"}:
                                # Handle union types
                                result[key] = [add_additional_properties_false(item) if isinstance(item, dict) else item for item in value]
                            elif isinstance(value, dict):
                                result[key] = add_additional_properties_false(value)
                            elif isinstance(value, list):
                                result[key] = [add_additional_properties_false(item) if isinstance(item, dict) else item for item in value]
                            else:
                                result[key] = value
                        return result
                    return obj

                schema_fixed = add_additional_properties_false(schema)
                create_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": model_name.lower(),
                        "strict": True,
                        "schema": schema_fixed,
                    }
                }
                logger.debug(f"Added structured output format for {model_name}")
            elif isinstance(response_format, dict):
                # Dict format - check if it's already in the right structure
                if "type" in response_format and "json_schema" in response_format:
                    # Format: {"type": "json_schema", "json_schema": {...}}
                    json_schema = response_format["json_schema"]
                    schema_name = json_schema["name"] if "name" in json_schema else "response"
                    schema_dict = json_schema["schema"] if "schema" in json_schema else {}
                    create_kwargs["text"] = {
                        "format": {
                            "type": response_format["type"],
                            "name": schema_name,
                            "strict": True,
                            "schema": {
                                **schema_dict,
                                "additionalProperties": False,
                            },
                        }
                    }
                else:
                    # Assume it's already in the correct format for text.format
                    create_kwargs["text"] = {"format": response_format}
                logger.debug("Added structured output format (dict)")

        # Add reasoning config
        reasoning = self._build_reasoning_config()
        if reasoning:
            create_kwargs["reasoning"] = reasoning  # pyright: ignore[reportArgumentType]

        # Add web search tools if enabled
        # Web search tools can be combined with function tools
        if self.config.enable_web_search:
            web_tool = {"type": "web_search"}
            if self.config.search_context_size:
                web_tool["search_context_size"] = self.config.search_context_size
            # Merge with existing tools
            if "tools" in create_kwargs:
                create_kwargs["tools"].append(web_tool)  # pyright: ignore[reportArgumentType]
            else:
                create_kwargs["tools"] = [web_tool]  # pyright: ignore[reportArgumentType]
            # Include web search sources when web search is enabled
            create_kwargs["include"] = ["web_search_call.action.sources"]  # pyright: ignore[reportArgumentType]

        logger.debug(f"Final create_kwargs: {create_kwargs}")
        if "text" in create_kwargs:
            logger.debug(f"text.format configured: {create_kwargs['text']}")

        # Generate request ID and store request before API call
        request_id = str(uuid4())
        request_data = {
            "model": self.config.model,
            "service_tier": self.config.service_tier,
            "reasoning_effort": self.config.reasoning_effort,
            "enable_web_search": self.config.enable_web_search,
            "search_context_size": self.config.search_context_size,
            "has_tools": tools is not None and len(tools) > 0,
            "tool_choice": str(tool_choice) if tool_choice else None,
            "has_response_format": response_format is not None,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Store request to DB before API call
        self._store_request_to_db(
            request_id=request_id,
            batch_id=None,  # Synchronous calls don't use batch_id
            response_id=None,  # Will be updated after response
            context=context or {},
            request_data=request_data,
            status="pending",
        )

        exception_info: tuple[type[BaseException], BaseException, Any] | None = None
        response: Any = None

        try:
            response = await self.client.responses.create(**create_kwargs)  # pyright: ignore[reportCallIssue]

            # Extract tool calls and output text from responses API
            tool_calls = []
            output_text = None

            # Responses API format - function calls are returned as type "function_call"
            if not hasattr(response, "output"):
                raise ValueError(f"Response has no output: {response}")

            for item in response.output:
                # Convert item to dict for easier handling
                #SafeMethodExc: LLM library has bad polymorphism - multiple item types
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    item_dict = {}

                item_type = item_dict["type"] if "type" in item_dict else ""

                if item_type == "function_call":
                    # Extract function call information
                    tc_data: dict[str, Any] = {
                        "call_id": item_dict["call_id"] if "call_id" in item_dict else None,
                        "name": item_dict["name"] if "name" in item_dict else None,
                    }
                    # Try to get arguments - might be a dict or string
                    arguments = item_dict["arguments"] if "arguments" in item_dict else None
                    if arguments:
                        if isinstance(arguments, str):
                            try:
                                tc_data["arguments"] = json.loads(arguments)
                            except json.JSONDecodeError:
                                tc_data["arguments"] = {}
                        else:
                            tc_data["arguments"] = arguments
                    else:
                        tc_data["arguments"] = {}
                    tool_calls.append(tc_data)

            output_text = response.output_text

            # Handle URL annotations - extract from response
            annotations = []
            for output in response.output:
                #SafeMethodExc: LLM library has bad polymorphism - content may not exist
                if not hasattr(output, "content") or output.content is None:
                    logger.warning(f"Output has no content: {output}")
                    continue

                for content in output.content:
                    annotations.extend(content.annotations)

            # Process URL citations from annotations
            url_citations = []
            for annotation in annotations:
                if annotation.type == "url_citation":
                    url = annotation.url
                    # Remove utm_source parameter if present
                    if url:
                        url = url.replace("?utm_source=chatgpt.com", "")
                        url = url.replace("?utm_source=openai", "")
                    title = getattr(annotation, "title", "")
                    start = getattr(annotation, "start_index", 0)
                    end = getattr(annotation, "end_index", 0)
                    reference_text = output_text[start:end] if output_text else ""

                    url_citations.append({
                        "url": url,
                        "title": title,
                        "start_index": start,
                        "end_index": end,
                        "reference_text": reference_text,
                    })

            # Update request in DB with response data
            response_dict = {
                "id": response.id,
                "status": getattr(response, "status", None),
                "output_text": output_text,
                "model": getattr(response, "model", None),
                "created_at": getattr(response, "created_at", None),
            }
            self._update_request_in_db(
                response_id=response.id,
                request_id=request_id,  # Also pass request_id for reference
                response_data=response_dict,
                usage=response.usage,
                status="completed",
            )

            # Track costs
            if response.usage:
                cost = self.calculate_cost(
                    response.usage, self.config.model, self.config.service_tier
                )
                self._total_cost += cost
                self._total_requests += 1

            return {
                "response": response,
                "output_text": output_text,
                "tool_calls": tool_calls,
                "usage": response.usage,
                "annotations": annotations,
                "url_citations": url_citations,
                "request_id": request_id,  # Include request_id in return for reference
            }

        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, OpenAIError) as e:
            # Log the exception (already being logged by the logger)
            logger.error(f"OpenAI API error in chat_with_tools: {type(e).__name__}: {e}", exc_info=True)
            # Store exception info for finally block
            exception_info = (type(e), e, e.__traceback__)
            raise
        finally:
            # Update request status if an exception occurred
            if exception_info:
                exc_type, exc_value, exc_traceback = exception_info
                error_message = str(exc_value)
                # Format error message for APIStatusError
                if isinstance(exc_value, APIStatusError):
                    error_message = f"{exc_value.status_code}: {error_message}"
                self._update_request_in_db(
                    request_id=request_id,
                    status="failed",
                    error_message=error_message,
                )
