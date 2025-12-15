"""
LLM agent for extracting structured vendor information from Prebid.js module files.

Analyzes Prebid.js BidAdapter, AnalyticsAdapter, RTD Provider, and other module files
to extract vendor names, domains, parameters, privacy support, and configuration details.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

from openai import AsyncOpenAI
from openai.types.responses import ResponseUsage
import yaml

from src.agent.data_models import PrebidVendorExtraction
from src.agent.gpt_client import BatchResult, OpenAIOrchestratorClient
from src.agent.llm_client import create_llm_client
from src.helper.config import load_app_config

logger = logging.getLogger(__name__)


class LlmClientProtocol(Protocol):
    """Protocol for LLM client interface that PrebidAgent requires."""

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
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
        response_id: str,
        response_data: dict[str, Any] | None = None,
        usage: ResponseUsage | None = None,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """Update request in database with response data."""
        ...

    @property
    def config_path(self) -> Path | None:
        """Path to config file."""
        ...

    @property
    def config(self) -> Any:
        """Client configuration."""
        ...


class PrebidAgent:
    """Agent for extracting vendor information from Prebid.js module files."""

    def __init__(self, config_path: Path) -> None:
        """
        Initialize the PrebidAgent.

        Args:
            config_path: Path to config file
        """
        config = load_app_config(config_path)
        llm_config = config.llm.prebid_extraction

        self.llm_client = create_llm_client(
            provider=llm_config.provider,
            model=llm_config.model_name,
            reasoning_effort=llm_config.reasoning_effort,
            enable_web_search=llm_config.enable_web_search,
            search_context_size=llm_config.search_context_size,
            service_tier=llm_config.service_tier,
            config_path=config_path,
            temperature=llm_config.temperature,
            max_iterations=1,  # Prebid processing is not iterative
        )

    def _load_prebid_extraction_prompt(self) -> str:
        """Load the extract_prebid_vendor prompt instructions from prompts.yaml."""
        if not self.llm_client.config_path:
            raise ValueError("config_path not set, cannot load prompts")
        prompts_path = self.llm_client.config_path.parent / "prompts.yaml"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
        with prompts_path.open("r", encoding="utf-8") as handle:
            prompts = yaml.safe_load(handle)
        return prompts.get("extract_prebid_vendor", {}).get("instructions", "")

    async def extract_prebid_vendor(
        self,
        vendor_name: str,
        module_type: str,
        js_content: str | None,
        md_content: str | None,
        background: bool | None = None,
    ) -> PrebidVendorExtraction:
        """
        Extract vendor information from Prebid.js module files.

        Args:
            vendor_name: Name of the vendor/module
            module_type: BidAdapter, RtdProvider, AnalyticsAdapter, IdSystem, or VideoProvider
            js_content: Content of .js or .ts file (or None)
            md_content: Content of .md documentation file (or None)
            background: Override config background setting. If None, uses config setting.
                       Note: Background mode requires using get_background_result() separately.

        Returns:
            PrebidVendorExtraction (raises ValueError if background=True, use extract_prebid_vendor_background instead)
        """
        use_background = (
            background if background is not None else self.llm_client.config.background
        )
        if use_background:
            raise ValueError(
                "extract_prebid_vendor() does not support background mode. "
                "Use extract_prebid_vendor_background() for background processing."
            )

        # Use chat_with_tools for synchronous extraction
        prompt_instructions = self._load_prebid_extraction_prompt()
        json_schema = PrebidVendorExtraction.model_json_schema()

        # Build user content with file contents
        user_content_parts = [
            f"Vendor name: {vendor_name}",
            f"Module type: {module_type}",
        ]
        if js_content:
            user_content_parts.append(
                f"\nJavaScript/TypeScript file content:\n{js_content}"
            )
        else:
            user_content_parts.append("\nJavaScript/TypeScript file: Not available")
        if md_content:
            user_content_parts.append(
                f"\nMarkdown documentation file content:\n{md_content}"
            )
        else:
            user_content_parts.append("\nMarkdown documentation file: Not available")

        user_content = "\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": user_content},
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "prebid_vendor_extraction",
                "schema": json_schema,
            },
        }

        result = await self.llm_client.chat_with_tools(
            messages=messages,
            tools=None,
            tool_choice=None,
            response_format=response_format,
        )

        output_text = result.get("output_text")
        if not output_text:
            raise ValueError("LLM response missing output text")

        return PrebidVendorExtraction.model_validate_json(output_text)

    async def extract_prebid_vendor_background(
        self,
        vendor_name: str,
        module_type: str,
        js_content: str | None,
        md_content: str | None,
        batch_id: str | None = None,
    ) -> BatchResult:
        """
        Submit Prebid vendor extraction request in background mode (for batch processing).

        Args:
            vendor_name: Name of the vendor/module
            module_type: BidAdapter, RtdProvider, AnalyticsAdapter, IdSystem, or VideoProvider
            js_content: Content of .js or .ts file (or None)
            md_content: Content of .md documentation file (or None)
            batch_id: Optional batch ID to group related requests

        Returns:
            BatchResult with response_id for later retrieval
        """
        # For background requests, we need to use the client's background API
        # Since we're using a protocol, we need to check if the client has background support
        # For now, we'll use a workaround by checking if the client has a responses.create method
        # This is a limitation - ideally the protocol would include background methods

        if not isinstance(self.llm_client, OpenAIOrchestratorClient):
            raise NotImplementedError(
                "Background extraction is only supported for OpenAIOrchestratorClient. "
                "Other clients need to implement background request support."
            )

        request_id = str(uuid4())
        if not batch_id:
            batch_id = str(uuid4())

        prompt_instructions = self._load_prebid_extraction_prompt()
        json_schema = PrebidVendorExtraction.model_json_schema()

        # Build user content with file contents
        user_content_parts = [
            f"Vendor name: {vendor_name}",
            f"Module type: {module_type}",
        ]
        if js_content:
            user_content_parts.append(
                f"\nJavaScript/TypeScript file content:\n{js_content}"
            )
        else:
            user_content_parts.append("\nJavaScript/TypeScript file: Not available")
        if md_content:
            user_content_parts.append(
                f"\nMarkdown documentation file content:\n{md_content}"
            )
        else:
            user_content_parts.append("\nMarkdown documentation file: Not available")

        user_content = "\n".join(user_content_parts)

        # For background requests, include response_format in the input messages
        # as the background API may not support top-level response_format
        system_message_with_format = (
            f"{prompt_instructions}\n\n"
            f"IMPORTANT: You must respond with valid JSON that matches this schema:\n"
            f"{json.dumps(json_schema, indent=2)}"
        )

        openai_client: AsyncOpenAI = self.llm_client.client  # type: ignore[attr-defined]
        model = self.llm_client.config.model  # type: ignore[attr-defined]
        service_tier = self.llm_client.config.service_tier  # type: ignore[attr-defined]
        reasoning_effort = self.llm_client.config.reasoning_effort  # type: ignore[attr-defined]

        # Build reasoning config
        reasoning: dict[str, Any] = {}
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort
        if self.llm_client.config.search_context_size == "high" or "gpt-5" in model.lower():  # type: ignore[attr-defined]
            reasoning["summary"] = "auto"
        reasoning_config = reasoning if reasoning else None

        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_message_with_format},
                {"role": "user", "content": user_content},
            ],
            "background": True,
            "service_tier": service_tier,
        }

        if reasoning_config:
            create_kwargs["reasoning"] = reasoning_config  # pyright: ignore[reportArgumentType]

        request_data = {
            "model": model,
            "service_tier": service_tier,
            "reasoning_effort": reasoning_effort,
            "enable_web_search": False,
            "search_context_size": "medium",
            "created_at": datetime.now(UTC).isoformat(),
        }

        try:
            response = await openai_client.responses.create(**create_kwargs)  # pyright: ignore[reportCallIssue]

            # Store request to DB using client's method
            self.llm_client._store_request_to_db(
                request_id=request_id,
                batch_id=batch_id,
                response_id=response.id,
                context={
                    "vendor_name": vendor_name,
                    "module_type": module_type,
                    "has_js": js_content is not None,
                    "has_md": md_content is not None,
                },
                request_data=request_data,
                status="pending",
            )

            return BatchResult(
                response_id=response.id,
                context={
                    "vendor_name": vendor_name,
                    "module_type": module_type,
                    "has_js": js_content is not None,
                    "has_md": md_content is not None,
                },
                request_data=request_data,
                batch_id=batch_id,
                request_id=request_id,
            )

        except Exception as e:
            error_msg = str(e)
            self.llm_client._store_request_to_db(
                request_id=request_id,
                batch_id=batch_id,
                response_id=None,
                context={
                    "vendor_name": vendor_name,
                    "module_type": module_type,
                    "has_js": js_content is not None,
                    "has_md": md_content is not None,
                },
                request_data=request_data,
                status="failed",
                error_message=error_msg,
            )
            logger.error(f"Error in extract_prebid_vendor_background: {e}", exc_info=True)
            raise

    async def extract_prebid_vendors_batch(
        self,
        vendors: list[dict[str, Any]],
        batch_id: str | None = None,
    ) -> list[BatchResult]:
        """
        Process multiple Prebid vendor extractions in batch (background mode).

        Args:
            vendors: List of vendor dictionaries with keys:
                - vendor_name: str
                - module_type: str
                - js_content: str | None
                - md_content: str | None
            batch_id: Optional batch ID to group related requests. If None, generates a new one.

        Returns:
            List of BatchResult objects with response IDs
        """
        if not batch_id:
            batch_id = str(uuid4())

        results = []
        for vendor in vendors:
            result = await self.extract_prebid_vendor_background(
                vendor_name=vendor["vendor_name"],
                module_type=vendor["module_type"],
                js_content=vendor.get("js_content"),
                md_content=vendor.get("md_content"),
                batch_id=batch_id,
            )
            results.append(result)

        return results

    async def get_prebid_background_result(
        self, response_id: str
    ) -> tuple[PrebidVendorExtraction | None, ResponseUsage | None, str]:
        """
        Retrieve the result of a Prebid vendor extraction background request.

        Args:
            response_id: The response ID from a background request

        Returns:
            Tuple of (PrebidVendorExtraction or None, usage or None, status)
        """

        if not isinstance(self.llm_client, OpenAIOrchestratorClient):
            raise NotImplementedError(
                "Background result retrieval is only supported for OpenAIOrchestratorClient. "
                "Other clients need to implement background result retrieval."
            )

        # Access the client's public client attribute
        openai_client: AsyncOpenAI = self.llm_client.client  # type: ignore[attr-defined]

        try:
            response = await openai_client.responses.retrieve(response_id)

            if response.status == "failed":
                logger.info(f"Background response {response_id} is failed")
                # Update DB status to reflect actual failed status
                self.llm_client._update_request_in_db(response_id, status="failed")
                return None, None, "failed"
            elif response.status != "completed":
                logger.info(
                    f"Background response {response_id} is still {response.status}"
                )
                # Update DB status
                self.llm_client._update_request_in_db(response_id, status="pending")
                return None, None, "pending"

            output = response.output_text
            if not output:
                logger.warning(
                    f"No output text in completed background response {response_id}"
                )
                self.llm_client._update_request_in_db(
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
            self.llm_client._update_request_in_db(
                response_id=response_id,
                response_data=response_dict,
                usage=response.usage,
                status="completed",
            )

            # Parse JSON output (may need to extract JSON from markdown code blocks)
            output_clean = output.strip()

            # Try to extract JSON from markdown code blocks
            if output_clean.startswith("```"):
                lines = output_clean.split("\n")
                json_lines = [
                    line for line in lines if not line.strip().startswith("```")
                ]
                output_clean = "\n".join(json_lines).strip()

            # Try to find JSON object in the output (handle cases where there's extra text)
            json_match = re.search(r"\{.*\}", output_clean, re.DOTALL)
            if json_match:
                output_clean = json_match.group(0)

            # Try to fix common JSON issues
            try:
                # Try parsing to validate JSON structure
                parsed = json.loads(output_clean)
                # Re-encode to ensure clean JSON
                output_clean = json.dumps(parsed)
            except json.JSONDecodeError:
                # If JSON is still invalid, try to fix common issues
                # Fix unescaped quotes in strings (basic attempt)
                output_clean = re.sub(r'(?<!\\)"(?=.*":)', '\\"', output_clean)
                # Try parsing again
                try:
                    parsed = json.loads(output_clean)
                    output_clean = json.dumps(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Could not fix JSON for response {response_id}: {e}"
                    )
                    # Store raw output for debugging
                    self.llm_client._update_request_in_db(
                        response_id,
                        status="failed",
                        error_message=f"Invalid JSON: {str(e)}",
                    )
                    return None, response.usage, "failed"

            extraction = PrebidVendorExtraction.model_validate_json(output_clean)
            return extraction, response.usage, "completed"

        except Exception as e:
            logger.error(
                f"Error retrieving Prebid background result {response_id}: {e}",
                exc_info=True,
            )
            self.llm_client._update_request_in_db(
                response_id, status="failed", error_message=str(e)
            )
            return None, None, "failed"
