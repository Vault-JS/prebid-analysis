"""
Prebid device storage disclosure URL extractor.

Extracts device storage disclosure using URLs identified in Prebid.js repo and checks
which disclosures are already stored in the database (loaded by TFC loader).

Usage:
    uv run python -m src.loaders.prebid_disclosure

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.helper.config import load_app_config
from src.helper.db_schema import DeviceStorageDisclosure
from src.helper.db_session import get_engine, get_session_factory


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


def check_disclosure_urls_in_db(session, urls: list[str]) -> dict[str, bool]:
    """
    Check which disclosure URLs exist in the device_storage_disclosures table.

    Returns a dictionary mapping URL -> exists (bool)
    """
    if not urls:
        return {}

    # Query all matching disclosure URLs
    results = (
        session.execute(
            select(DeviceStorageDisclosure.disclosure_url).where(
                DeviceStorageDisclosure.disclosure_url.in_(urls)
            )
        )
        .scalars()
        .all()
    )

    found_urls = set(results)

    # Build result dictionary
    return {url: url in found_urls for url in urls}


def process_prebid_disclosures(modules_path: Path) -> dict[str, Any]:
    """
    Process all JSON files in the Prebid.js metadata/modules directory
    and check disclosure URLs against the database.

    Returns a summary dictionary with:
    - total_files: number of JSON files processed
    - files_with_disclosures: number of files that have disclosures field
    - total_urls_found: total number of disclosure URLs found
    - urls_in_db: number of URLs found in database
    - urls_missing: number of URLs not found in database
    - missing_urls: list of URLs not found in database
    - file_results: per-file breakdown
    """
    if not modules_path.exists():
        raise FileNotFoundError(f"Modules directory not found: {modules_path}")

    json_files = list(modules_path.glob("*.json"))

    total_urls_found = 0
    urls_in_db_count = 0
    urls_missing_count = 0
    missing_urls: list[str] = []
    files_with_disclosures = 0
    file_results: list[dict[str, Any]] = []

    config = load_app_config()
    engine = get_engine(config)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                urls = extract_disclosure_urls(data)
                total_urls_found += len(urls)

                file_result = {
                    "file": json_file.name,
                    "urls_found": len(urls),
                    "urls": urls,
                }

                if urls:
                    files_with_disclosures += 1
                    url_status = check_disclosure_urls_in_db(session, urls)

                    found_count = sum(1 for exists in url_status.values() if exists)
                    missing_count = len(urls) - found_count

                    urls_in_db_count += found_count
                    urls_missing_count += missing_count

                    file_result["urls_in_db"] = found_count
                    file_result["urls_missing"] = missing_count
                    file_result["url_status"] = url_status

                    # Collect missing URLs
                    for url, exists in url_status.items():
                        if not exists:
                            missing_urls.append(url)
                else:
                    file_result["urls_in_db"] = 0
                    file_result["urls_missing"] = 0
                    file_result["url_status"] = {}

                file_results.append(file_result)

            except (json.JSONDecodeError, IOError, OSError) as e:
                print(f"Error processing {json_file.name}: {e}")
                file_results.append(
                    {
                        "file": json_file.name,
                        "error": str(e),
                    }
                )

    return {
        "total_files": len(json_files),
        "files_with_disclosures": files_with_disclosures,
        "total_urls_found": total_urls_found,
        "urls_in_db": urls_in_db_count,
        "urls_missing": urls_missing_count,
        "missing_urls": sorted(set(missing_urls)),  # Deduplicate and sort
        "file_results": file_results,
    }


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary of the disclosure URL checking results."""
    print("\n" + "=" * 80)
    print("Prebid Disclosure URL Check Summary")
    print("=" * 80)
    print(f"Total JSON files processed: {results['total_files']}")
    print(f"Files with disclosures: {results['files_with_disclosures']}")
    print(f"Total disclosure URLs found: {results['total_urls_found']}")
    print(f"URLs found in database: {results['urls_in_db']}")
    print(f"URLs missing from database: {results['urls_missing']}")

    if results["missing_urls"]:
        print("\nMissing URLs:")
        for url in results["missing_urls"]:
            print(f"  - {url}")

    print("\n" + "=" * 80)


def main() -> None:
    """Main entry point."""
    config = load_app_config()

    # Construct path to Prebid.js metadata/modules directory
    modules_path = (
        config.ingestion.prebid_temp_dir
        / config.ingestion.prebid_extract_dirname
        / "metadata"
        / "modules"
    )

    if not modules_path.exists():
        print(f"Warning: Modules directory not found: {modules_path}")
        print("You may need to run the Prebid loader first to download the repository.")
        return

    results = process_prebid_disclosures(modules_path)
    print_summary(results)


if __name__ == "__main__":
    main()
