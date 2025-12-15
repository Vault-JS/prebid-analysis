"""
Prebid vendor extraction processor.

Processes PrebidVendor records using PrebidAgent to extract structured vendor
information. Handles incremental processing based on file hash changes and tracks
LLM request status.

Usage:
    uv run python -m src.agent.process_prebid_code [--config CONFIG_PATH] [--force-reprocess] [--limit LIMIT]

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
import argparse
import asyncio
import logging
from pathlib import Path

from sqlalchemy import select

from src.agent.prebid_agent import PrebidAgent
from src.helper.config import load_app_config
from src.helper.db_operations import (
    get_latest_prebid_vendor_extraction,
    get_pending_llm_request_for_vendor,
    store_prebid_vendor_extraction,
    transactional_session,
    update_prebid_vendor_hashes,
)
from src.helper.db_schema import PrebidVendor, PrebidVendorExtraction
from src.helper.db_session import get_engine, get_session_factory
from src.loaders.prebid_hash_utils import (
    get_vendor_file_hashes,
    has_vendor_files_changed,
)


# Log first few vendors for debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_file_content(file_path: Path) -> str | None:
    """Read file content, returning None if file doesn't exist."""
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (IOError, OSError):
        return None


def get_vendors_to_process(
    session, modules_path: Path, force_reprocess: bool = False
) -> list[PrebidVendor]:
    """
    Get list of vendors that need processing.

    Args:
        session: Database session
        modules_path: Base path to Prebid.js modules directory
        force_reprocess: If True, process all vendors regardless of file changes

    Returns:
        List of PrebidVendor records that need processing
    """
    all_vendors = list(session.execute(select(PrebidVendor)).scalars().all())
    vendors_to_process = []

    for vendor in all_vendors:
        if force_reprocess:
            vendors_to_process.append(vendor)
            continue

        # Check if files have changed
        current_hashes = get_vendor_file_hashes(vendor, modules_path)
        files_changed = has_vendor_files_changed(vendor, current_hashes)

        if files_changed:
            # Files changed - needs reprocessing
            vendors_to_process.append(vendor)
        else:
            # Files haven't changed - check if we need to process
            latest_extraction = get_latest_prebid_vendor_extraction(session, vendor.id)
            if latest_extraction is None:
                # No extraction yet - we need to process this vendor
                # It might have a pending request (to check if it completed)
                # Or it might need a new request submitted
                vendors_to_process.append(vendor)

    return vendors_to_process


async def process_vendor(
    vendor: PrebidVendor,
    modules_path: Path,
    config_path: Path,
    prebid_agent: PrebidAgent,
    verbose: bool = True,
    collect_only: bool = False,
) -> tuple[bool, str]:
    """
    Process a single vendor: check for existing request, submit if needed, retrieve result once.

    Args:
        vendor: PrebidVendor record to process
        modules_path: Base path to Prebid.js modules directory
        config_path: Path to config file
        prebid_agent: PrebidAgent instance for extraction
        verbose: If True, log detailed information

    Returns:
        Tuple of (success: bool, message: str)
    """
    logger = logging.getLogger(__name__)

    if verbose:
        logger.info(f"Processing vendor: {vendor.vendor_name} ({vendor.module_type})")

    # Check if there's already a completed extraction
    with transactional_session(config_path) as session:
        latest_extraction = get_latest_prebid_vendor_extraction(session, vendor.id)
        if verbose and latest_extraction:
            logger.info(f"  Found existing extraction for {vendor.vendor_name}")

    # Check if files have changed
    current_hashes = get_vendor_file_hashes(vendor, modules_path)
    files_changed = has_vendor_files_changed(vendor, current_hashes)

    if verbose:
        logger.info(f"  Files changed: {files_changed}")
        js_hash = current_hashes.get("js")
        logger.info(f"  Current hashes: js={js_hash[:16] if js_hash else 'None'}...")
        logger.info(
            f"  Stored hashes: js={vendor.js_file_hash[:16] if vendor.js_file_hash else 'None'}..."
        )

    # If we have a completed extraction and files haven't changed, skip
    if latest_extraction and not files_changed:
        if verbose:
            logger.info(
                f"  Skipping {vendor.vendor_name} - already extracted and files unchanged"
            )
        return True, "already_extracted"

    # Read file contents
    js_content = None
    md_content = None

    if vendor.js_file_path:
        js_path = modules_path.parent / vendor.js_file_path
        js_content = read_file_content(js_path)

    if vendor.ts_file_path:
        ts_path = modules_path.parent / vendor.ts_file_path
        js_content = read_file_content(ts_path)  # Use TS if available, otherwise JS

    if vendor.md_file_path:
        md_path = modules_path.parent / vendor.md_file_path
        md_content = read_file_content(md_path)

    # If files changed, always submit a new request (don't use existing one)
    if files_changed:
        if collect_only:
            if verbose:
                logger.info(
                    f"  Skipping submission for {vendor.vendor_name} (files changed) - collect only mode"
                )
            return False, "skipped_collect_only"

        result = await prebid_agent.extract_prebid_vendor_background(
            vendor_name=vendor.vendor_name,
            module_type=vendor.module_type,
            js_content=js_content,
            md_content=md_content,
        )
        # Update file hashes immediately after submission to prevent resubmission
        with transactional_session(config_path) as session:
            update_prebid_vendor_hashes(
                session,
                vendor.id,
                current_hashes.get("js"),
                current_hashes.get("md"),
                current_hashes.get("ts"),
            )
            session.commit()
        if verbose:
            logger.info(f"  Updated hashes after submission for {vendor.vendor_name}")
        return False, f"submitted:{result.response_id}"

    # Check if there's an existing request for this vendor (only if files haven't changed)
    with transactional_session(config_path) as session:
        existing_request = get_pending_llm_request_for_vendor(
            session, vendor.vendor_name, vendor.module_type
        )
        if verbose:
            if existing_request:
                logger.info(
                    f"  Found existing request: {existing_request.response_id}, status: {existing_request.status}"
                )
            else:
                logger.info(f"  No existing request found for {vendor.vendor_name}")

    response_id: str | None = None

    # If there's an existing request (pending or completed), try to retrieve it once
    if existing_request and existing_request.response_id:
        if verbose:
            logger.info(
                f"  Attempting to retrieve result for {existing_request.response_id}"
            )
        response_id = existing_request.response_id

        # Check if extraction already exists for this request
        with transactional_session(config_path) as session:
            existing_extraction = session.execute(
                select(PrebidVendorExtraction).where(
                    PrebidVendorExtraction.llm_request_id == response_id
                )
            ).scalar_one_or_none()

        # If extraction already exists for this request, skip (already processed)
        if existing_extraction:
            return (
                False,
                "pending",
            )  # Treat as pending since we're waiting for file changes

        decision, usage, status = await prebid_agent.get_prebid_background_result(response_id)

        # If completed, store the result
        if status == "completed" and decision is not None:
            # Store extraction result
            with transactional_session(config_path) as session:
                extraction_dict = decision.model_dump()
                store_prebid_vendor_extraction(
                    session,
                    {
                        "prebid_vendor_id": vendor.id,
                        "llm_request_id": response_id,
                        "vendor_name": extraction_dict["vendor_name"],
                        "product_name": extraction_dict.get("product_name"),
                        "bidder_code": extraction_dict.get("bidder_code"),
                        "maintainer": extraction_dict.get("maintainer"),
                        "gdpr_supported": extraction_dict["privacy_support"]["gdpr"],
                        "gpp_supported": extraction_dict["privacy_support"]["gpp"],
                        "ccpa_usp_supported": extraction_dict["privacy_support"][
                            "ccpa_usp"
                        ],
                        "coppa_supported": extraction_dict["privacy_support"]["coppa"],
                        "schain_supported": extraction_dict["privacy_support"][
                            "schain"
                        ],
                        "eids_supported": extraction_dict["privacy_support"]["eids"],
                        "floors_supported": extraction_dict["features"]["floors_supported"],
                        "app_supported": extraction_dict["features"]["app_supported"],
                        "s2s_supported": extraction_dict["features"]["s2s_supported"],
                        "user_ids": extraction_dict["features"]["user_ids"],
                        "gvlid": extraction_dict.get("gvlid"),
                        "supported_media_types": extraction_dict.get(
                            "supported_media_types"
                        ),
                        "currency": extraction_dict.get("currency"),
                        "extracted_data": extraction_dict,
                        "extraction_notes": extraction_dict.get("extraction_notes"),
                        "stage": "llm_linked",
                    },
                )
                session.commit()

            # Update file hashes
            current_hashes = get_vendor_file_hashes(vendor, modules_path)
            with transactional_session(config_path) as session:
                update_prebid_vendor_hashes(
                    session,
                    vendor.id,
                    current_hashes.get("js"),
                    current_hashes.get("md"),
                    current_hashes.get("ts"),
                )
                session.commit()

            return True, "completed"
        elif status == "pending":
            return False, "pending"
        elif status == "failed":
            # Request failed - resubmit it
            if collect_only:
                if verbose:
                    logger.info(
                        f"  Request {response_id} failed, skipping resubmission - collect only mode"
                    )
                return False, "failed_skipped_resubmit"

            if verbose:
                logger.info(f"  Request {response_id} failed, resubmitting...")
            result = await prebid_agent.extract_prebid_vendor_background(
                vendor_name=vendor.vendor_name,
                module_type=vendor.module_type,
                js_content=js_content,
                md_content=md_content,
            )
            # Update file hashes immediately after submission to prevent resubmission
            with transactional_session(config_path) as session:
                update_prebid_vendor_hashes(
                    session,
                    vendor.id,
                    current_hashes.get("js"),
                    current_hashes.get("md"),
                    current_hashes.get("ts"),
                )
                session.commit()
            if verbose:
                logger.info(f"  Resubmitted as {result.response_id}")
            return False, f"resubmitted:{result.response_id}"
        else:
            return False, f"unknown_status:{status}"
    else:
        # No existing request, submit a new one
        if collect_only:
            if verbose:
                logger.info(
                    f"  No existing request for {vendor.vendor_name}, skipping submission - collect only mode"
                )
            return False, "skipped_collect_only"

        result = await prebid_agent.extract_prebid_vendor_background(
            vendor_name=vendor.vendor_name,
            module_type=vendor.module_type,
            js_content=js_content,
            md_content=md_content,
        )
        # Update file hashes immediately after submission to prevent resubmission
        with transactional_session(config_path) as session:
            update_prebid_vendor_hashes(
                session,
                vendor.id,
                current_hashes.get("js"),
                current_hashes.get("md"),
                current_hashes.get("ts"),
            )
            session.commit()
        if verbose:
            logger.info(f"  Updated hashes after submission for {vendor.vendor_name}")
        return False, f"submitted:{result.response_id}"


async def process_vendors_batch(
    vendors: list[PrebidVendor],
    modules_path: Path,
    config_path: Path,
    limit: int | None = None,
    collect_only: bool = False,
) -> None:
    """
    Process vendors: check for existing requests, submit if needed, retrieve results once.

    Args:
        vendors: List of PrebidVendor records to process
        modules_path: Base path to Prebid.js modules directory
        config_path: Path to config file
        limit: Limit the number of vendors to process
        collect_only: If True, only collect results, do not submit new requests
    """
    prebid_agent = PrebidAgent(config_path)

    total_vendors = len(vendors)
    logger.info(f"Processing {limit} or total {total_vendors} vendors...")

    submitted_count = 0
    completed_count = 0
    pending_count = 0
    already_extracted_count = 0
    failed_count = 0
    resubmitted_count = 0

    VENDOR_LOG_INTERVAL = 10
    VERBOSE_VENDOR_COUNT = 5
    for i, vendor in enumerate(vendors, 1):
        if limit is not None and i > limit:
            logger.info(f"Limit reached: {limit}")
            break
        if i % VENDOR_LOG_INTERVAL == 0:
            logger.info(f"Processed {i}/{total_vendors} vendors...")

        success, message = await process_vendor(
            vendor,
            modules_path,
            config_path,
            prebid_agent,
            verbose=(i <= VERBOSE_VENDOR_COUNT),
            collect_only=collect_only,
        )

        if i <= VERBOSE_VENDOR_COUNT:
            logger.info(
                f"Vendor {i}: {vendor.vendor_name} ({vendor.module_type}) -> {message}"
            )

        if success and message == "already_extracted":
            already_extracted_count += 1
        elif success and message == "completed":
            completed_count += 1
        elif message.startswith("submitted:"):
            submitted_count += 1
        elif message.startswith("resubmitted:"):
            resubmitted_count += 1
        elif message == "pending":
            pending_count += 1
        else:
            failed_count += 1

    logger.info("\nProcessing summary:")
    logger.info(f"  Already extracted (up to date): {already_extracted_count}")
    logger.info(f"  Completed (retrieved from existing requests): {completed_count}")
    logger.info(f"  Submitted (new requests): {submitted_count}")
    logger.info(f"  Resubmitted (failed requests restarted): {resubmitted_count}")
    logger.info(f"  Pending (waiting for completion): {pending_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"\nTotal vendors: {total_vendors}")
    logger.info(
        f"\nNote: {submitted_count + resubmitted_count + pending_count} vendors are in progress. "
        "Run this script again later to check for completed results."
    )


def extract_prebid_vendors(
    force_reprocess: bool = False, limit: int | None = None, collect_only: bool = False
) -> None:
    """
    Main function to extract Prebid vendor information using LLM.

    Args:
        force_reprocess: If True, reprocess all vendors even if files haven't changed
        limit: Limit the number of vendors to process
        collect_only: If True, only collect results, do not submit new requests
    """
    config = load_app_config()
    engine = get_engine(config)
    session_factory = get_session_factory(engine)

    # Get Prebid.js modules path
    modules_path = (
        config.ingestion.prebid_temp_dir
        / config.ingestion.prebid_extract_dirname
        / "modules"
    )

    if not modules_path.exists():
        raise FileNotFoundError(
            f"Prebid.js modules directory not found: {modules_path}. "
            "Please run prebid_loader first to download the repository."
        )

    # Get vendors that need processing
    with session_factory() as session:
        vendors_to_process = get_vendors_to_process(
            session, modules_path, force_reprocess
        )

    if not vendors_to_process:
        logger.info("No vendors need processing. All vendors are up to date.")
        return

    logger.info(f"Found {len(vendors_to_process)} vendors to process")

    # Process vendors using async batch processing
    # Process vendors using async batch processing
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    asyncio.run(
        process_vendors_batch(
            vendors_to_process, modules_path, config_path, limit, collect_only
        )
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Prebid vendor information using LLM processing."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all vendors, even if files haven't changed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of vendors to process",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Only collect results from existing requests, do not submit new ones",
    )
    args = parser.parse_args()

    extract_prebid_vendors(
        force_reprocess=args.force, limit=args.limit, collect_only=args.collect
    )


if __name__ == "__main__":
    main()
