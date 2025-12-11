"""
Prebid.js vendor data loader.

Downloads Prebid.js repository from GitHub and extracts vendor information from
module files (BidAdapters, RtdProviders, AnalyticsAdapters, IdSystems, VideoProviders).
Tracks file hashes to detect changes for incremental updates.

Usage:
    uv run python -m src.loaders.prebid_loader

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from datetime import UTC, date, datetime, timezone
import io
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4
import zipfile

import requests
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.helper.config import load_app_config
from src.helper.db_schema import (
    DeviceStorageDisclosure,
    PrebidVendor,
    create_all_tables,
)
from src.helper.db_session import get_engine, get_session_factory
from src.loaders.prebid_hash_utils import compute_file_hash


# Module type patterns - these match the file naming conventions
MODULE_TYPE_PATTERNS = {
    "BidAdapter": re.compile(r"(.+?)BidAdapter\.(js|ts)$", re.IGNORECASE),
    "RtdProvider": re.compile(r"(.+?)RtdProvider\.(js|ts)$", re.IGNORECASE),
    "AnalyticsAdapter": re.compile(r"(.+?)AnalyticsAdapter\.(js|ts)$", re.IGNORECASE),
    "IdSystem": re.compile(r"(.+?)IdSystem\.(js|ts)$", re.IGNORECASE),
    "VideoProvider": re.compile(r"(.+?)VideoProvider\.(js|ts)$", re.IGNORECASE),
}


def download_prebid_repo(repo_url: str, target_dir: Path, timeout_seconds: int) -> Path:
    """Download and extract Prebid.js repository from GitHub."""
    print(f"Downloading Prebid.js from {repo_url}")
    target_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(repo_url, timeout=timeout_seconds)
    response.raise_for_status()

    archive = io.BytesIO(response.content)
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.namelist():
            zip_file.extract(member, path=target_dir)
    print(f"Repository extracted into {target_dir}")
    return target_dir


def extract_module_type_and_name(filename: str) -> tuple[str | None, str | None]:
    """Extract module type and vendor name from filename."""
    # Try matching as-is first (for .js and .ts files)
    for module_type, pattern in MODULE_TYPE_PATTERNS.items():
        match = pattern.match(filename)
        if match:
            vendor_name = match.group(1)
            return module_type, vendor_name

    # If no match and file is .md, try matching by replacing .md with .js
    if filename.lower().endswith(".md"):
        test_filename = filename[:-3] + ".js"
        for module_type, pattern in MODULE_TYPE_PATTERNS.items():
            match = pattern.match(test_filename)
            if match:
                vendor_name = match.group(1)
                return module_type, vendor_name

    return None, None


def collect_vendors(modules_path: Path) -> list[dict[str, Any]]:
    """Collect all vendor modules from the Prebid.js modules directory."""
    vendors: list[dict[str, Any]] = []
    vendor_files: dict[str, dict[str, str | None]] = {}  # vendor_name -> {js, md, ts}

    # First pass: collect all module files
    for file_path in modules_path.rglob("*"):
        if not file_path.is_file():
            continue

        filename = file_path.name
        module_type, vendor_name = extract_module_type_and_name(filename)

        if module_type and vendor_name:
            # Normalize vendor name (handle camelCase, etc.)
            vendor_key = vendor_name.lower()

            if vendor_key not in vendor_files:
                vendor_files[vendor_key] = {
                    "vendor_name": vendor_name,
                    "module_type": module_type,
                    "js": None,
                    "md": None,
                    "ts": None,
                }

            # Store file paths relative to modules directory
            rel_path = str(file_path.relative_to(modules_path.parent))
            if filename.endswith(".js"):
                vendor_files[vendor_key]["js"] = rel_path
            elif filename.endswith(".ts"):
                vendor_files[vendor_key]["ts"] = rel_path
            elif filename.endswith(".md"):
                vendor_files[vendor_key]["md"] = rel_path

        # Second pass: create vendor records with metadata and hashes
    for _vendor_key, file_info in vendor_files.items():
        # Get file metadata and hashes
        raw_metadata: dict[str, Any] = {}
        js_path = file_info.get("js")
        md_path = file_info.get("md")
        ts_path = file_info.get("ts")
        js_hash: str | None = None
        md_hash: str | None = None
        ts_hash: str | None = None

        if js_path:
            full_js_path = modules_path.parent / js_path
            if full_js_path.exists():
                raw_metadata["js_size"] = full_js_path.stat().st_size
                raw_metadata["js_lines"] = count_lines(full_js_path)
                js_hash = compute_file_hash(full_js_path)

        if ts_path:
            full_ts_path = modules_path.parent / ts_path
            if full_ts_path.exists():
                raw_metadata["ts_size"] = full_ts_path.stat().st_size
                raw_metadata["ts_lines"] = count_lines(full_ts_path)
                ts_hash = compute_file_hash(full_ts_path)

        if md_path:
            full_md_path = modules_path.parent / md_path
            if full_md_path.exists():
                raw_metadata["md_size"] = full_md_path.stat().st_size
                raw_metadata["md_lines"] = count_lines(full_md_path)
                md_hash = compute_file_hash(full_md_path)

        vendors.append(
            {
                "vendor_name": file_info["vendor_name"],
                "module_type": file_info["module_type"],
                "js_file_path": js_path,
                "md_file_path": md_path,
                "ts_file_path": ts_path,
                "js_file_hash": js_hash,
                "md_file_hash": md_hash,
                "ts_file_hash": ts_hash,
                "raw_metadata": raw_metadata if raw_metadata else None,
            }
        )

    return vendors


def count_lines(file_path: Path) -> int:
    """Count lines in a file, handling encoding errors gracefully."""
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except (IOError, OSError):
        return 0


def refresh_repository(
    temp_dir: Path, extract_dirname: str, repo_url: str, timeout_seconds: int
) -> Path:
    """Refresh the Prebid.js repository by downloading fresh copy."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    download_prebid_repo(repo_url, temp_dir, timeout_seconds)
    extracted_path = temp_dir / extract_dirname
    if not extracted_path.exists():
        raise FileNotFoundError(f"Extracted Prebid.js path not found: {extracted_path}")
    return extracted_path


def upsert_vendors(
    session, vendors: list[dict[str, Any]], retrieved_at: datetime
) -> None:
    """Upsert vendor records into the database."""
    for vendor in vendors:
        # Create a unique ID
        vendor_id = str(uuid4())

        stmt = sqlite_insert(PrebidVendor).values(
            id=vendor_id,
            vendor_name=vendor["vendor_name"],
            module_type=vendor["module_type"],
            js_file_path=vendor.get("js_file_path"),
            md_file_path=vendor.get("md_file_path"),
            ts_file_path=vendor.get("ts_file_path"),
            js_file_hash=vendor.get("js_file_hash"),
            md_file_hash=vendor.get("md_file_hash"),
            ts_file_hash=vendor.get("ts_file_hash"),
            raw_metadata=vendor.get("raw_metadata"),
            extracted_data=None,  # Will be populated by LLM processing
            stage="raw",
            retrieved_at=retrieved_at,
        )
        # Use the unique index on (vendor_name, module_type) for conflict resolution
        stmt = stmt.on_conflict_do_update(
            index_elements=["vendor_name", "module_type"],
            set_={
                "js_file_path": vendor.get("js_file_path"),
                "md_file_path": vendor.get("md_file_path"),
                "ts_file_path": vendor.get("ts_file_path"),
                "js_file_hash": vendor.get("js_file_hash"),
                "md_file_hash": vendor.get("md_file_hash"),
                "ts_file_hash": vendor.get("ts_file_hash"),
                "raw_metadata": vendor.get("raw_metadata"),
                "retrieved_at": retrieved_at,
                "updated_at": datetime.now(UTC),
            },
        )
        session.execute(stmt)


def extract_disclosure_urls(data: dict[str, Any]) -> list[str]:
    """
    Extract all disclosure URLs from a JSON file's data structure.

    The disclosures field is a dict where:
    - Keys are the disclosure URLs (must start with http:// or https://)
    - Values are objects with metadata

    Returns a list of all found URLs.
    """
    urls: list[str] = []

    # Check if disclosures field exists
    if "disclosures" not in data:
        return urls

    disclosures = data["disclosures"]

    # Handle None or empty dict
    if not disclosures:
        return urls

    # disclosures should be a dict where keys are URLs
    if isinstance(disclosures, dict):
        for key in disclosures.keys():
            if isinstance(key, str):
                key_stripped = key.strip()
                # Simple check: URL must start with http:// or https://
                if key_stripped.startswith("http://") or key_stripped.startswith(
                    "https://"
                ):
                    urls.append(key_stripped)

    return urls


def load_device_storage_disclosure(url: str) -> dict[str, Any] | None:
    """
    Load device storage disclosure data from vendor URL.
    Returns the full JSON data structure or None if failed.
    """
    try:
        print(f"  Fetching device storage disclosure from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("  Successfully loaded device storage disclosure")
        return data
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  [!] Failed to load device storage disclosure: {e}")
        return None


def process_prebid_disclosures(
    session: Session, modules_path: Path, collection_date: date
) -> None:
    """
    Process all JSON files in the Prebid.js metadata/modules directory,
    extract disclosure URLs, and update/insert device storage disclosures.
    """
    metadata_modules_path = modules_path.parent / "metadata" / "modules"

    if not metadata_modules_path.exists():
        print(f"Warning: Metadata modules directory not found: {metadata_modules_path}")
        return

    json_files = list(metadata_modules_path.glob("*.json"))

    total_urls_found = 0
    urls_updated = 0
    urls_inserted = 0
    urls_failed = 0

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            urls = extract_disclosure_urls(data)
            total_urls_found += len(urls)

            if not urls:
                continue

            # Extract vendor name from filename (without .json extension)
            prebid_vendor_name = json_file.stem

            for disclosure_url in urls:
                # Check if URL already exists in database (can be multiple rows)
                existing_disclosures = list(
                    session.execute(
                        select(DeviceStorageDisclosure).where(
                            DeviceStorageDisclosure.disclosure_url == disclosure_url
                        )
                    )
                    .scalars()
                    .all()
                )

                if existing_disclosures:
                    # Update all existing records with prebid_vendor_name
                    updated_count = 0
                    for existing_disclosure in existing_disclosures:
                        if existing_disclosure.prebid_vendor_name != prebid_vendor_name:
                            existing_disclosure.prebid_vendor_name = prebid_vendor_name
                            updated_count += 1
                    if updated_count > 0:
                        urls_updated += updated_count
                        print(
                            f"  Updated prebid_vendor_name for {updated_count} record(s) with URL {disclosure_url}: {prebid_vendor_name}"
                        )
                else:
                    # Fetch disclosure data and insert new record
                    disclosure_data = load_device_storage_disclosure(disclosure_url)

                    disclosure_record = {
                        "id": str(uuid4()),
                        "tcf_2v2_vendor_id": None,  # Empty for Prebid entries
                        "disclosure_url": disclosure_url,
                        "prebid_vendor_name": prebid_vendor_name,
                        "collection_date": collection_date,
                        "fetched_at": None,
                        "raw_response": None,
                        "disclosures": None,
                        "domains": None,
                        "error_message": None,
                    }

                    if disclosure_data:
                        disclosure_record["raw_response"] = disclosure_data
                        disclosure_record["disclosures"] = disclosure_data.get(
                            "disclosures", []
                        )
                        disclosure_record["domains"] = disclosure_data.get(
                            "domains", []
                        )
                        disclosure_record["fetched_at"] = datetime.now(UTC)
                        urls_inserted += 1
                        print(
                            f"  Inserted new disclosure for {prebid_vendor_name}: {disclosure_url}"
                        )
                    else:
                        disclosure_record["error_message"] = (
                            "Failed to fetch disclosure data"
                        )
                        urls_failed += 1
                        print(
                            f"  Failed to fetch disclosure for {prebid_vendor_name}: {disclosure_url}"
                        )

                    # Insert new record
                    stmt = sqlite_insert(DeviceStorageDisclosure).values(
                        **disclosure_record
                    )
                    session.execute(stmt)

        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Error processing {json_file.name}: {e}")

    print("\n[*] Prebid disclosure processing:")
    print(f"    Total disclosure URLs found: {total_urls_found}")
    print(f"    URLs updated: {urls_updated}")
    print(f"    URLs inserted: {urls_inserted}")
    print(f"    URLs failed: {urls_failed}")


def ingest_prebid() -> None:
    """Main ingestion function for Prebid.js vendor data."""
    config = load_app_config()
    engine = get_engine(config)
    create_all_tables(engine)
    session_factory = get_session_factory(engine)

    extracted_path = refresh_repository(
        config.ingestion.prebid_temp_dir,
        config.ingestion.prebid_extract_dirname,
        config.ingestion.prebid_repo_url,
        config.ingestion.prebid_download_timeout_seconds,
    )

    modules_path = extracted_path / "modules"
    if not modules_path.exists():
        raise FileNotFoundError(f"Prebid.js modules directory missing: {modules_path}")

    vendors = collect_vendors(modules_path)
    retrieved_at = datetime.now(timezone.utc)
    collection_date = date.today()

    with session_factory() as session:
        upsert_vendors(session, vendors, retrieved_at)
        session.commit()

    print(f"Stored {len(vendors)} Prebid.js vendors in the database.")

    # Process disclosure URLs from metadata/modules JSON files
    print("\n[*] Processing Prebid disclosure URLs...")
    with session_factory() as session:
        process_prebid_disclosures(session, modules_path, collection_date)
        session.commit()


def main() -> None:
    """Main entry point."""
    ingest_prebid()


if __name__ == "__main__":
    main()
