"""
Database operations for creating and updating canonical entities.

Provides functions for upserting Company, BrandOrProduct, Domain, DomainOrLink,
CompanyRelationship, and Evidence records. Includes transactional session management
and helper functions for querying LLM requests and vendor extractions.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.helper.db_schema import (
    LlmRequest,
    PrebidVendor,
    PrebidVendorExtraction,
    PrebidDoc
)
from src.helper.config import default_config_path, load_config_cached
from src.helper.db_session import get_engine, get_session_factory

STAGE_DEFAULT = "llm_linked"


@lru_cache(maxsize=None)
def _session_factory(config_path: str):
    config = load_config_cached(config_path)
    engine = get_engine(config)
    return get_session_factory(engine)


def reset_caches() -> None:
    load_config_cached.cache_clear()
    _session_factory.cache_clear()


@contextmanager
def get_session(config_path: Path | None = None) -> Iterator[Session]:
    path = str((config_path or default_config_path()).resolve())
    factory = _session_factory(path)
    with factory() as session:
        yield session


@contextmanager
def transactional_session(config_path: Path | None = None) -> Iterator[Session]:
    with get_session(config_path) as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise


def _merge_json(target: dict | list | None, incoming: dict | list | None) -> Any:
    if incoming is None:
        return target
    return incoming

def store_llm_request(session: Session, data: dict[str, Any]) -> LlmRequest:
    request_id = data["id"]
    request = session.get(LlmRequest, request_id)
    if request is None:
        request = LlmRequest(id=request_id)
        session.add(request)
    request.batch_id = data.get("batch_id", request.batch_id)
    request.response_id = data.get("response_id", request.response_id)
    request.model = data.get("model", request.model)
    request.service_tier = data.get("service_tier", request.service_tier)
    request.status = data.get("status", request.status or "pending")
    request.context = data.get("context", request.context)
    request.request_data = data.get("request_data", request.request_data)
    request.response_data = data.get("response_data", request.response_data)
    request.cost = data.get("cost", request.cost)
    request.input_tokens = data.get("input_tokens", request.input_tokens)
    request.output_tokens = data.get("output_tokens", request.output_tokens)
    request.reasoning_tokens = data.get("reasoning_tokens", request.reasoning_tokens)
    request.error_message = data.get("error_message", request.error_message)
    request.completed_at = data.get("completed_at", request.completed_at)
    return request


def get_llm_request_by_response_id(
    session: Session, response_id: str
) -> LlmRequest | None:
    return session.execute(
        select(LlmRequest).where(LlmRequest.response_id == response_id)
    ).scalar_one_or_none()


def get_llm_requests_by_batch_id(session: Session, batch_id: str) -> list[LlmRequest]:
    return list(
        session.execute(
            select(LlmRequest)
            .where(LlmRequest.batch_id == batch_id)
            .order_by(LlmRequest.created_at)
        )
        .scalars()
        .all()
    )


def get_pending_llm_request_for_vendor(
    session: Session, vendor_name: str, module_type: str
) -> LlmRequest | None:
    """
    Get pending LLM request for a vendor if one exists.
    Also checks for completed requests that might need result retrieval.

    Args:
        session: Database session
        vendor_name: Vendor name
        module_type: Module type

    Returns:
        LlmRequest with pending or completed status, or None if not found
    """
    # Filter requests by checking JSON fields directly in SQL
    requests = list(
        session.execute(
            select(LlmRequest)
            .where(LlmRequest.status.in_(["pending", "completed"]))
            .order_by(LlmRequest.created_at.desc())
        )
        .scalars()
        .all()
    )

    # Filter in Python (SQLite JSON extraction can be slow, so we do it in Python)
    for request in requests:
        context = request.context or {}
        if (
            context.get("vendor_name") == vendor_name
            and context.get("module_type") == module_type
        ):
            return request

    return None


def get_pending_only_llm_request_for_vendor(
    session: Session, vendor_name: str, module_type: str
) -> LlmRequest | None:
    """
    Get ONLY pending LLM request for a vendor (not completed ones).

    Args:
        session: Database session
        vendor_name: Vendor name
        module_type: Module type

    Returns:
        LlmRequest with pending status, or None if not found
    """
    requests = list(
        session.execute(
            select(LlmRequest)
            .where(LlmRequest.status == "pending")
            .order_by(LlmRequest.created_at.desc())
        )
        .scalars()
        .all()
    )

    for request in requests:
        context = request.context or {}
        if (
            context.get("vendor_name") == vendor_name
            and context.get("module_type") == module_type
        ):
            return request

    return None


def store_prebid_vendor_extraction(
    session: Session, data: dict[str, Any]
) -> PrebidVendorExtraction:
    """
    Store a Prebid vendor extraction result.

    Args:
        session: Database session
        data: Dictionary with extraction data including:
            - id: UUID (optional, generated if not provided)
            - prebid_vendor_id: Required
            - llm_request_id: Optional
            - vendor_name: Required
            - product_name: Optional
            - bidder_code: Optional
            - maintainer: Optional
            - gdpr_supported, gpp_supported, etc.: Privacy flags
            - gvlid: Optional
            - supported_media_types: Optional list
            - currency: Optional
            - extracted_data: Required full JSON
            - extraction_confidence: Optional
            - extraction_notes: Optional
            - stage: Optional (defaults to "raw")
            - extracted_at: Optional (defaults to now)

    Returns:
        PrebidVendorExtraction record
    """
    extraction_id = data.get("id") or str(uuid.uuid4())
    extraction = PrebidVendorExtraction(
        id=extraction_id,
        prebid_vendor_id=data["prebid_vendor_id"],
        llm_request_id=data.get("llm_request_id"),
        vendor_name=data["vendor_name"],
        product_name=data.get("product_name"),
        bidder_code=data.get("bidder_code"),
        maintainer=data.get("maintainer"),
        gdpr_supported=data.get("gdpr_supported", False),
        gpp_supported=data.get("gpp_supported", False),
        ccpa_usp_supported=data.get("ccpa_usp_supported", False),
        coppa_supported=data.get("coppa_supported", False),
        schain_supported=data.get("schain_supported", False),
        eids_supported=data.get("eids_supported", False),
        gvlid=data.get("gvlid"),
        supported_media_types=data.get("supported_media_types"),
        currency=data.get("currency"),
        extracted_data=data["extracted_data"],
        extraction_confidence=data.get("extraction_confidence"),
        extraction_notes=data.get("extraction_notes"),
        stage=data.get("stage", "raw"),
        extracted_at=data.get("extracted_at", datetime.now(UTC)),
    )
    session.add(extraction)
    return extraction


def get_prebid_vendor_extractions(
    session: Session, prebid_vendor_id: str
) -> list[PrebidVendorExtraction]:
    """
    Get all extraction records for a vendor (history).

    Args:
        session: Database session
        prebid_vendor_id: Prebid vendor ID

    Returns:
        List of extractions ordered by extracted_at descending (newest first)
    """
    return list(
        session.execute(
            select(PrebidVendorExtraction)
            .where(PrebidVendorExtraction.prebid_vendor_id == prebid_vendor_id)
            .order_by(PrebidVendorExtraction.extracted_at.desc())
        )
        .scalars()
        .all()
    )


def get_latest_prebid_vendor_extraction(
    session: Session, prebid_vendor_id: str
) -> PrebidVendorExtraction | None:
    """
    Get the most recent extraction for a vendor.

    Args:
        session: Database session
        prebid_vendor_id: Prebid vendor ID

    Returns:
        Most recent extraction or None
    """
    return session.execute(
        select(PrebidVendorExtraction)
        .where(PrebidVendorExtraction.prebid_vendor_id == prebid_vendor_id)
        .order_by(PrebidVendorExtraction.extracted_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def update_prebid_vendor_hashes(
    session: Session,
    vendor_id: str,
    js_hash: str | None,
    md_hash: str | None,
    ts_hash: str | None,
) -> None:
    """
    Update file hashes for a Prebid vendor.

    Args:
        session: Database session
        vendor_id: Prebid vendor ID
        js_hash: JS file hash (or None)
        md_hash: MD file hash (or None)
        ts_hash: TS file hash (or None)
    """
    vendor = session.get(PrebidVendor, vendor_id)
    if vendor:
        if js_hash is not None:
            vendor.js_file_hash = js_hash
        if md_hash is not None:
            vendor.md_file_hash = md_hash
        if ts_hash is not None:
            vendor.ts_file_hash = ts_hash

