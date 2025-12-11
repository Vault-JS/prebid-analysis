"""
Prebid file hash utilities.

Computes SHA256 hashes for Prebid vendor module files (JS, MD, TS) and compares
them with stored hashes to detect file changes for incremental processing.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
import hashlib
from pathlib import Path

from src.helper.db_schema import PrebidVendor


def compute_file_hash(file_path: Path) -> str | None:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of SHA256 hash, or None if file doesn't exist or error occurs
    """
    if not file_path.exists() or not file_path.is_file():
        return None

    try:
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except (IOError, OSError):
        return None


def get_vendor_file_hashes(
    vendor: PrebidVendor, modules_path: Path
) -> dict[str, str | None]:
    """
    Compute current file hashes for a vendor's files.

    Args:
        vendor: PrebidVendor record
        modules_path: Base path to Prebid.js modules directory

    Returns:
        Dictionary with keys 'js', 'md', 'ts' and hash values (or None)
    """
    hashes: dict[str, str | None] = {"js": None, "md": None, "ts": None}

    if vendor.js_file_path:
        js_path = modules_path.parent / vendor.js_file_path
        hashes["js"] = compute_file_hash(js_path)

    if vendor.md_file_path:
        md_path = modules_path.parent / vendor.md_file_path
        hashes["md"] = compute_file_hash(md_path)

    if vendor.ts_file_path:
        ts_path = modules_path.parent / vendor.ts_file_path
        hashes["ts"] = compute_file_hash(ts_path)

    return hashes


def has_vendor_files_changed(
    vendor: PrebidVendor, current_hashes: dict[str, str | None]
) -> bool:
    """
    Check if vendor files have changed by comparing current hashes with stored hashes.

    Args:
        vendor: PrebidVendor record with stored hashes
        current_hashes: Dictionary with current file hashes ('js', 'md', 'ts')

    Returns:
        True if any file has changed. Returns False if no hashes are stored (use pending request/extraction check instead).
    """
    # If vendor has no stored hashes, we can't determine if files changed
    # Return False - caller should check for pending requests/extractions instead
    if not vendor.js_file_hash and not vendor.md_file_hash and not vendor.ts_file_hash:
        return False

    # Compare each file hash
    if vendor.js_file_path and vendor.js_file_hash != current_hashes.get("js"):
        return True

    if vendor.md_file_path and vendor.md_file_hash != current_hashes.get("md"):
        return True

    if vendor.ts_file_path and vendor.ts_file_hash != current_hashes.get("ts"):
        return True

    return False
