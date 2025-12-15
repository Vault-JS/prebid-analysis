import logging
import csv
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

from sqlalchemy import select
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from src.helper.config import load_app_config
from src.helper.db_schema import PrebidDoc, PrebidVendorExtraction, PrebidVendor
from src.helper.db_session import get_engine, get_session_factory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: List[Any], y_pred: List[Any], label: str) -> None:
    """
    Calculate and print metrics for binary, categorical, or set data.
    """
    clean_true = []
    clean_pred = []

    # Filter out None values to ensure valid comparison
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None:
            clean_true.append(t)
            clean_pred.append(p)

    if not clean_true:
        print(f"--- {label} ---")
        print("No valid overlapping data points found.")
        print()
        return

    # Check for complex types (list, tuple, set) which sklearn doesn't handle well for accuracy
    is_complex = any(isinstance(x, (list, tuple, set)) for x in clean_true)

    # Manual accuracy (Exact Match)
    manual_matches = sum(1 for t, p in zip(clean_true, clean_pred) if t == p)
    manual_acc = manual_matches / len(clean_true) if clean_true else 0.0

    print(f"--- {label} ---")
    print(f"Count: {len(clean_true)}")

    # if not is_complex:
    #     try:
    #          acc = accuracy_score(clean_true, clean_pred)
    #          print(f"Accuracy (sklearn): {acc:.2%}")
    #     except ValueError:
    #          print("Accuracy (sklearn): N/A (ValueError)")

    print(f"Accuracy (manual):  {manual_acc:.2%}")

    # Check if boolean for F1 and Confusion Matrix
    is_bool = all(isinstance(x, bool) or x in (0, 1) for x in clean_true) and \
              all(isinstance(x, bool) or x in (0, 1) for x in clean_pred)

    if is_bool and not is_complex:
        f1 = f1_score(clean_true, clean_pred, zero_division=0)
        print(f"F1 Score: {f1:.2f}")

        try:
            tn, fp, fn, tp = confusion_matrix(clean_true, clean_pred, labels=[False, True]).ravel()
            print(f"Confusion Matrix:")
            print(f"       LLM: False | LLM: True")
            print(f"Doc: False  {tn:^5} | {fp:^5}")
            print(f"Doc: True   {fn:^5} | {tp:^5}")
        except ValueError:
             # Handle case where confusion_matrix might not return 4 values
             cm = confusion_matrix(clean_true, clean_pred)
             print("Confusion Matrix (Labels might vary):")
             print(cm)
    elif is_complex:
        print("(Complex type comparison: Exact match only)")
        # Show sample disagreements for complex types
        disagreements = [(t, p) for t, p in zip(clean_true, clean_pred) if t != p]
        if disagreements:
            print(f"Disagreements: {len(disagreements)}")
            print(f"Sample Disagreement: Doc={disagreements[0][0]} vs Extracted={disagreements[0][1]}")
    else:
        # For non-boolean scalar, maybe show top disagreements?
        disagreements = [(t, p) for t, p in zip(clean_true, clean_pred) if t != p]
        if disagreements:
            print(f"Disagreements: {len(disagreements)}")

    print()


def prepare_llm_comparisons(docs: List[PrebidDoc], extractions: List[PrebidVendorExtraction]) -> Dict[str, Dict[str, List[Any]]]:
    """
    Prepares data vectors for comparison between PrebidDoc and PrebidVendorExtraction.
    """
    # Create a mapping from prebid_vendor_id to PrebidDoc for easier lookup
    doc_map = {doc.prebid_vendor_id: doc for doc in docs if doc.extracted_metadata}

    # Comparison Config
    comparisons_config = [
        # Boolean Flags
        ("GDPR Supported", "tcfeu_supported", "gdpr_supported"),
        ("USP / CCPA Supported", "usp_supported", "ccpa_usp_supported"),
        ("COPPA Supported", "coppa_supported", "coppa_supported"),
        ("Schains Supported", "schain_supported", "schain_supported"),
        ("Floors Supported", "floors_supported", "floors_supported"),
        ("S2S Supported (PBS)", "pbs", "s2s_supported"),
        # Values
        ("GPP Supported", "gpp_supported", "gpp_supported"),
        ("App Supported", "pbs_app_supported", "app_supported"),
        # Values
        ("Bidder Code", "biddercode", "bidder_code"),
        # Complex
        ("Media Types", "media_types", "supported_media_types"),
        ("User IDs", "userIds", "user_ids"),
    ]

    data_vectors = {label: {"doc": [], "extracted": []} for label, _, _ in comparisons_config}

    for extraction in extractions:
        doc = doc_map.get(extraction.prebid_vendor_id)
        if not doc:
            continue # Skip if no corresponding doc with metadata

        doc_meta = doc.extracted_metadata or {}

        for label, doc_key, extract_attr in comparisons_config:
            doc_val = doc_meta.get(doc_key)
            extract_val = getattr(extraction, extract_attr, None)

            # Normalization
            if label == "Bidder Code":
                if isinstance(doc_val, str): doc_val = doc_val.lower()
                if isinstance(extract_val, str): extract_val = extract_val.lower()

            elif label == "Media Types":
                # Normalize to sorted tuple for comparison
                if isinstance(doc_val, str):
                    # "banner, video" -> {"banner", "video"}
                    doc_val = sorted(list({x.strip().lower() for x in doc_val.split(",") if x.strip()}))
                elif isinstance(doc_val, list):
                    doc_val = sorted(list({str(x).lower() for x in doc_val}))
                else:
                    doc_val = []

                if isinstance(extract_val, list):
                    extract_val = sorted(list({str(x).lower() for x in extract_val}))
                else:
                    extract_val = []

                # Tuple for hashing/equality
                doc_val = tuple(doc_val)
                extract_val = tuple(extract_val)

            elif label == "User IDs":
                # Normalize user IDs (list of strings or boolean 'all'/'none' sometimes?)
                # Doc "userIds" can be boolean (False) or a list/string?
                # Based on user input: "userIds": false.
                # Let's normalize to a set of strings or None/False.
                # If boolean False, treat as empty set.
                if isinstance(doc_val, bool) and not doc_val:
                    doc_val = set()
                elif isinstance(doc_val, str):
                    if doc_val.lower() == "all":
                         # Special case: "all" - maybe keep as special token or ignore comparison?
                         # For now, let's treat "all" as a set containing "all"
                         doc_val = {"ALL"}
                    else:
                        doc_val = set(x.strip().lower() for x in doc_val.split(",") if x.strip())
                elif isinstance(doc_val, list):
                     doc_val = set(str(x).lower() for x in doc_val)
                else:
                    doc_val = set()

                if isinstance(extract_val, list):
                    extract_val = set(str(x).lower() for x in extract_val)
                elif isinstance(extract_val, bool) and not extract_val:
                    extract_val = set()
                else:
                    extract_val = set()

                # If extracted is None -> empty set
                if extract_val is None: extract_val = set()

            # Let's Normalize booleans to False if None
            if label in ["GDPR Supported", "USP / CCPA Supported", "COPPA Supported",
                         "Schains Supported", "Floors Supported", "S2S Supported (PBS)",
                         "GPP Supported", "App Supported"]:
                doc_val = bool(doc_val)
                extract_val = bool(extract_val)

            data_vectors[label]["doc"].append(doc_val)
            data_vectors[label]["extracted"].append(extract_val)

    return data_vectors


def load_csv_data(filepath: Path) -> dict[str, dict[str, Any]]:
    """Load bidder data from CSV into a dict keyed by bidder-code."""
    data = {}
    if not filepath.exists():
        logger.warning(f"CSV file not found: {filepath}")
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("bidder-code", "").strip().lower()
            if code:
                data[code] = row
    return data


def normalize_csv_bool(value: Any) -> bool:
    """Normalize CSV 'yes'/'no' etc. to boolean."""
    if isinstance(value, bool):
        return value
    if not value:
        return False
    v = str(value).strip().lower()
    return v in ("yes", "true", "1")


def compare_csv_vs_doc(doc_records: list[PrebidDoc], csv_data: dict[str, dict[str, Any]]):
    """Compare PrebidDoc metadata against CSV data."""
    print("\n" + "=" * 80)
    print("COMPARISON: PrebidDoc vs CSV Data")
    print("=" * 80)

    comparisons = {
        "media_types": {"matches": 0, "total": 0, "mismatches": []},
        "schain_supported": {"matches": 0, "total": 0, "mismatches": []},
        "tcfeu_supported": {"matches": 0, "total": 0, "mismatches": []},
        "usp_supported": {"matches": 0, "total": 0, "mismatches": []},
        "coppa_supported": {"matches": 0, "total": 0, "mismatches": []},
        "floors_supported": {"matches": 0, "total": 0, "mismatches": []},
        "gpp_supported": {"matches": 0, "total": 0, "mismatches": []},
        "dchain_supported": {"matches": 0, "total": 0, "mismatches": []},
        "multiformat_supported": {"matches": 0, "total": 0, "mismatches": []},
        "safeframes_ok": {"matches": 0, "total": 0, "mismatches": []},
        "deals_supported": {"matches": 0, "total": 0, "mismatches": []},
        "ortb_blocking_supported": {"matches": 0, "total": 0, "mismatches": []},
        # User IDs handled separately due to complexity? Or straightforward string match if normalized?
        # CSV user-ids: "britepoolId  criteo ..." (space separated?) or comma?
        # Let's check CSV format for user-ids.
    }

    # Iterate over Doc records as the base
    for doc in doc_records:
        if not doc.extracted_metadata:
            continue

        meta = doc.extracted_metadata
        bidder_code = meta.get("biddercode")
        if not bidder_code:
            continue

        bidder_code = str(bidder_code).strip().lower()
        if bidder_code not in csv_data:
            continue

        csv_row = csv_data[bidder_code]
        title = meta.get("title", bidder_code)

        # --- 1. Media Types Comparison ---
        doc_media_raw = meta.get("media_types", "")
        if isinstance(doc_media_raw, str):
            doc_media = set(
                t.strip().lower() for t in doc_media_raw.split(",") if t.strip()
            )
        elif isinstance(doc_media_raw, list):
            doc_media = set(str(t).strip().lower() for t in doc_media_raw if t)
        else:
            doc_media = set()

        csv_media = set()
        if normalize_csv_bool(csv_row.get("banner")):
            csv_media.add("banner")
        if normalize_csv_bool(csv_row.get("video")):
            csv_media.add("video")
        if normalize_csv_bool(csv_row.get("native")):
            csv_media.add("native")

        comparisons["media_types"]["total"] += 1
        if doc_media == csv_media:
            comparisons["media_types"]["matches"] += 1
        else:
            comparisons["media_types"]["mismatches"].append(
                f"{title}: Doc={doc_media} != CSV={csv_media}"
            )



        # --- 2. User IDs Comparison ---
        # Doc: "userIds": "all" or ["id1", "id2"] or False
        # CSV: "user-ids": "all" or "none" or "id1  id2"
        comparisons.setdefault("user_ids", {"matches": 0, "total": 0, "mismatches": []})

        doc_ids_raw = meta.get("userIds")
        # Normalize Doc
        doc_ids = set()
        if isinstance(doc_ids_raw, str):
            if doc_ids_raw.lower() == "all":
                doc_ids = {"ALL"}
            else:
                 # CSV seems to use spaces, but Doc might use commas?
                 # Let's support both comma and space splitting
                parts = re.split(r'[,\s]+', doc_ids_raw)
                doc_ids = set(p.strip().lower() for p in parts if p.strip())
        elif isinstance(doc_ids_raw, list):
            doc_ids = set(str(x).lower() for x in doc_ids_raw)

        # Normalize CSV
        csv_ids_raw = csv_row.get("user-ids", "").lower().strip()
        csv_ids = set()
        if csv_ids_raw == "all":
            csv_ids = {"ALL"}
        elif csv_ids_raw and csv_ids_raw != "none":
            # Split by double space or single space
            parts = re.split(r'\s+', csv_ids_raw)
            csv_ids = set(p.strip() for p in parts if p.strip())

        comparisons["user_ids"]["total"] += 1
        if doc_ids == csv_ids:
            comparisons["user_ids"]["matches"] += 1
        else:
             # Can be noisy, so maybe stricter logging?
             if len(doc_ids) > 0 or len(csv_ids) > 0:
                 comparisons["user_ids"]["mismatches"].append(
                    f"{title}: Doc={doc_ids_raw} != CSV={csv_ids}"
                )

        # --- 3. Boolean Flags Comparison ---
        bool_map = {
            "schain_supported": ("schain_supported", "schain"),
            "tcfeu_supported": ("tcfeu_supported", "tcfeu"),
            "usp_supported": ("usp_supported", "usp"),
            "coppa_supported": ("coppa_supported", "coppa"),
            "floors_supported": ("floors_supported", "floors"),
            "gpp_supported": ("gpp_supported", "gpp"),
            "dchain_supported": ("dchain_supported", "dchain"),
            "multiformat_supported": ("multiformat_supported", "multiformat"),
            "safeframes_ok": ("safeframes_ok", "safeframes"),
            "deals_supported": ("deals_supported", "deals"),
            "ortb_blocking_supported": ("ortb_blocking_supported", "ortb-blocking"),
        }

        for metric, (doc_key, csv_col) in bool_map.items():
            doc_val = normalize_csv_bool(meta.get(doc_key, False))
            csv_val = normalize_csv_bool(csv_row.get(csv_col))

            comparisons[metric]["total"] += 1
            if doc_val == csv_val:
                comparisons[metric]["matches"] += 1
            else:
                comparisons[metric]["mismatches"].append(
                    f"{title}: Doc={meta.get(doc_key, False)} != CSV={csv_val} (RowVal='{csv_row.get(csv_col)}')"
                )

    # Print Results
    for metric, data in comparisons.items():
        total = data["total"]
        if total == 0:
            print(f"{metric:<20}: No data found for comparison.")
            continue

        accuracy = (data["matches"] / total) * 100
        print(f"{metric:<20}: Accuracy: {accuracy:.2f}% ({data['matches']}/{total})")

        if data["mismatches"]:
            print("  Top 5 Mismatches:")
            for m in data["mismatches"][:5]:
                print(f"    - {m}")


def main():
    config = load_app_config()
    engine = get_engine(config)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        # 1. Fetch Data
        print("Fetching data...")
        extractions = session.execute(select(PrebidVendorExtraction)).scalars().all()
        docs = session.execute(select(PrebidDoc)).scalars().all()

        # Load CSV
        csv_path = Path("data/bidder-data.csv")
        csv_data = load_csv_data(csv_path)
        print(f"Loaded {len(extractions)} extractions, {len(docs)} docs, {len(csv_data)} CSV rows.")

        # 2. Compare Doc vs LLM (Original)
        print("\n" + "=" * 80)
        print("COMPARISON: PrebidDoc vs LLM Extraction")
        print("=" * 80)
        data_vectors = prepare_llm_comparisons(docs, extractions)
        for label in data_vectors:
            calculate_metrics(data_vectors[label]["doc"], data_vectors[label]["extracted"], label)

        # 3. Compare Doc vs CSV (New)
        compare_csv_vs_doc(docs, csv_data)


if __name__ == "__main__":
    main()
