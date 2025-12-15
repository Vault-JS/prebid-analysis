"""
Loader for prebid.github.io documentation data.

Downloads the repo, iterates over markdown files in dev-docs/bidders,
extracts YAML frontmatter, and stores in prebid_docs table.
"""
from __future__ import annotations
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4
import zipfile

import requests
import yaml
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from src.helper.config import load_app_config
from src.helper.db_schema import PrebidDoc, PrebidVendor
from src.helper.db_session import get_engine, get_session_factory
from src.loaders.prebid_hash_utils import compute_file_hash

logger = logging.getLogger(__name__)

def download_docs_repo(repo_url: str, target_dir: Path, timeout_seconds: int) -> Path:
    """Download and extract prebid.github.io repository."""
    logger.info(f"Downloading prebid.github.io from {repo_url}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(repo_url, timeout=timeout_seconds)
    response.raise_for_status()

    archive = io.BytesIO(response.content)
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.namelist():
            zip_file.extract(member, path=target_dir)
    logger.info(f"Repository extracted into {target_dir}")
    return target_dir


def parse_doc_file(file_path: Path) -> tuple[dict[str, Any], str]:
    """
    Parse aekyll markdown file with YAML frontmatter.
    Returns (metadata_dict, content_str).
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split by the first two '---' markers
    # YAML frontmatter is commonly between --- and ---
    parts = re.split(r"^---$", content, maxsplit=2, flags=re.MULTILINE)

    metadata = {}
    markdown_body = content

    if len(parts) >= 3:
        # parts[0] is usually empty (before first ---)
        # parts[1] is the YAML
        # parts[2] is the rest
        try:
            metadata = yaml.safe_load(parts[1])
            markdown_body = parts[2]
        except yaml.YAMLError as e:
            logger.warning(f"  Warning: YAML parse error in {file_path.name}: {e}")

    return metadata, markdown_body


def upsert_docs(session: Session, docs_path: Path, extract_root: Path) -> None:
    """Iterate docs and upsert into DB."""

    # Path to bidders docs: dev-docs/bidders/
    bidders_path = docs_path / "dev-docs" / "bidders"

    if not bidders_path.exists():
        logger.warning(f"Warning: Bidders docs path not found: {bidders_path}")
        return

    count = 0
    # Pre-fetch existing vendors to minimize queries
    # Map (bidder_code) -> vendor_id
    # Also Map (vendor_name) -> vendor_id (fallback)

    # Optimization: Loading all BidAdapters
    vendors = session.execute(
        select(PrebidVendor.id, PrebidVendor.vendor_name, PrebidVendor.raw_metadata)
        .where(PrebidVendor.module_type == 'BidAdapter')
    ).all()

    # Create simple lookup maps
    # Note: raw_metadata isn't always populated with 'bidder_code' yet since that comes from LLM
    # So we rely on:
    # 1. vendor_name matching
    # 2. filename matching (slug)

    vendor_map = {row.vendor_name.lower(): row.id for row in vendors}

    for file_path in bidders_path.glob("*.md"):
        slug = file_path.stem  # e.g. "criteo"

        metadata, content = parse_doc_file(file_path)
        file_hash = compute_file_hash(file_path)

        # Determining Vendor Link
        # 1. 'biddercode' in metadata
        # 2. filename slug matching vendor_name

        prebid_vendor_id = None
        bidder_code = metadata.get('biddercode')

        if bidder_code and isinstance(bidder_code, str):
            # Try to match key in vendor_map
            # Often biddercode == vendor_name
            prebid_vendor_id = vendor_map.get(bidder_code.lower())

        if not prebid_vendor_id:
             # Fallback to slug
             prebid_vendor_id = vendor_map.get(slug.lower())

        # Insert/Update
        doc_id = str(uuid4())

        stmt = sqlite_insert(PrebidDoc).values(
            id=doc_id,
            prebid_vendor_id=prebid_vendor_id,
            slug=slug,
            file_path=str(file_path.relative_to(extract_root)),
            raw_content=content,
            file_hash=file_hash,
            extracted_metadata=metadata,
            retrieved_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Upsert on slug (unique index)
        stmt = stmt.on_conflict_do_update(
            index_elements=['slug'],
            set_={
                "prebid_vendor_id": prebid_vendor_id,
                "file_path": str(file_path.relative_to(extract_root)),
                "raw_content": content,
                "file_hash": file_hash,
                "extracted_metadata": metadata,
                "retrieved_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
        )

        session.execute(stmt)
        count += 1

    logger.info(f"Processed {count} documentation files.")


def print_ingestion_stats(session: Session) -> None:
    """Print statistics about vendor ingestion and documentation coverage."""
    # 1. Total Code Vendors (BidAdapters)
    code_vendors_count = session.execute(
        select(PrebidVendor.id).where(PrebidVendor.module_type == 'BidAdapter')
    ).all()
    total_code_vendors = len(code_vendors_count)

    # 2. Total Docs
    total_docs = session.query(PrebidDoc).count()

    # 3. Intersection (Vendors with Docs)
    # Docs where prebid_vendor_id is not null
    linked_docs_count = session.query(PrebidDoc).where(PrebidDoc.prebid_vendor_id.is_not(None)).count()

    # Docs Intersection is same as Vendors Intersection (assuming 1:1 mostly, but one vendor can have multiple docs? unlikely)
    # Let's count distinct vendor IDs in docs to be safe
    linked_vendors_count = session.query(PrebidDoc.prebid_vendor_id).where(PrebidDoc.prebid_vendor_id.is_not(None)).distinct().count()

    # 4. Missing in Code (Orphan Docs)
    orphan_docs = session.query(PrebidDoc.slug).where(PrebidDoc.prebid_vendor_id.is_(None)).all()
    orphan_docs_count = len(orphan_docs)
    orphan_docs = [row.slug for row in orphan_docs]

    # 5. Missing in Docs (Code vendors without Docs)
    # Using a subquery for efficiency or set difference in python
    # All vendor IDs
    all_vendor_ids = {row.id for row in code_vendors_count}
    # Linked vendor IDs
    linked_vendor_ids = {row.prebid_vendor_id for row in session.query(PrebidDoc.prebid_vendor_id).where(PrebidDoc.prebid_vendor_id.is_not(None)).all()}

    missing_docs_ids = all_vendor_ids - linked_vendor_ids
    missing_docs_count = len(missing_docs_ids)

    # Get names for sample
    missing_docs = []
    if missing_docs_ids:
        missing_docs = session.execute(
            select(PrebidVendor.vendor_name).where(PrebidVendor.id.in_(list(missing_docs_ids)))
        ).scalars().all()

    logger.info("\n" + "="*50)
    logger.info("INGESTION STATISTICS")
    logger.info("="*50)
    logger.info(f"Total Code Vendors (BidAdapters): {total_code_vendors}")
    logger.info(f"Total Documentation Files:        {total_docs}")
    logger.info("-" * 30)
    logger.info(f"Matched (linked_docs_count):      {linked_docs_count}")
    logger.info(f"Matched (linked_vendors_count):   {linked_vendors_count}")
    logger.info("-" * 30)
    logger.info(f"Missing in Code (probaly renamed/deleted):   {orphan_docs_count}")
    if orphan_docs:
        logger.info(f"  Sample: {', '.join(orphan_docs)}...")

    logger.info(f"Missing in Docs (undocumented modules):      {missing_docs_count}")
    if missing_docs:
        logger.info(f"  Sample: {', '.join(missing_docs)}...")
    logger.info("="*50 + "\n")


def ingest_prebid_docs() -> None:
    """Main function to ingest prebid docs."""
    config = load_app_config()
    engine = get_engine(config)
    session_factory = get_session_factory(engine)

    # 1. Download
    extract_root = download_docs_repo(
        config.ingestion.prebid_docs_repo_url,
        Path(config.ingestion.prebid_temp_dir) / "docs_repo",
        config.ingestion.prebid_download_timeout_seconds
    )

    # The zip usually extracts to a folder name based on branch, e.g. prebid.github.io-master
    # But download_docs_repo extracts contents INTO target_dir via my implementation
    # Wait, my implementation of download_prebid_repo extracts INTO target_dir?
    # Let's check download_prebid_repo in prebid_loader.py to match behavior.
    # It extracts member, so it creates a subdir usually.

    # Adjusting path: The config says `prebid.github.io-master`
    docs_root = extract_root / config.ingestion.prebid_docs_extract_dirname

    if not docs_root.exists():
        # Fallback: list dirs and pick first
        subdirs = [d for d in extract_root.iterdir() if d.is_dir()]
        if subdirs:
            docs_root = subdirs[0]
            print(f"  Assuming docs root is {docs_root}")
        else:
            print(f"Error: Could not find extracted docs root in {extract_root}")
            return

    # 2. Process
    with session_factory() as session:
        upsert_docs(session, docs_root, extract_root)
        print_ingestion_stats(session)
        session.commit()

if __name__ == "__main__":
    ingest_prebid_docs()
