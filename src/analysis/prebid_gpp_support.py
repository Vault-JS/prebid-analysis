#!/usr/bin/env python3
"""
Prebid GPP Support Analysis.

Checks if Prebid vendors actually implement GPP (Global Privacy Platform) support
by checking their endpoints using data extracted in the database.

Replaces src/analysis/prebid_gpp_support_old.py.
"""
import asyncio
import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import aiohttp
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.helper.config import load_app_config
from src.helper.db_session import get_engine, get_session_factory
from src.helper.db_schema import PrebidVendor, PrebidVendorExtraction, PrebidDoc

# GPP Strings (from old analysis)
GPP_STRINGS = {
    "eu": "DBABMA~CPXxRfAPXxRfAAfKABENB-CgAAAAAAAAAAYgAAAAAAAA",
    "us-california": "DBABBg~BUoAAABg.Q",
    "us-virginia": "DBABNA~B-9Q",
    "us": "DBABLA~B-7Q",
}

@dataclass
class VendorData:
    vendor_name: str
    doc_gpp_supported: Any  # From PrebidDoc.extracted_metadata
    extraction_gpp_supported: bool  # From PrebidVendorExtraction.gpp_supported
    urls: List[str]

@dataclass
class EndpointResult:
    endpoint: str
    results: Dict[str, Dict[str, Any]]  # region -> {success, status, url}
    all_flags_supported: bool
    at_least_one_flag_supported: bool

@dataclass
class VendorResult:
    vendor_name: str
    vendor_data: Dict[str, Any]
    endpoint_results: List[EndpointResult]
    # Summary stats for this vendor
    all_endpoints_gpp: bool  # All endpoints support GPP (based on criteria below)
    one_endpoint_gpp: bool   # At least one endpoint supports GPP

    # We need to aggregating logic for "All endpoints support GPP" vs "At least one..."
    # calculated over the intersection with "All regions" vs "One region"

    # Storing raw results makes it easier to aggregate later
    raw_results: List[Dict[str, Any]]

async def check_url(
    session: aiohttp.ClientSession,
    url: str,
    sem: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Checks a single URL against all GPP strings.
    """
    results = {}

    # We'll return a dict keyed by region
    async with sem:
        for region, gpp_str in GPP_STRINGS.items():
            if "?" in url:
                test_url = f"{url}&gpp={gpp_str}"
            else:
                test_url = f"{url}?gpp={gpp_str}"

            try:
                async with session.get(test_url, timeout=10) as response:
                    status = response.status
                    # Consider 200-399 as success
                    success = 200 <= status < 400
                    results[region] = {
                        "test_url": test_url,
                        "status_code": status,
                        "success": success
                    }
            except Exception as e:
                results[region] = {
                    "test_url": test_url,
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }

    return results

def get_vendor_data(session: Session) -> List[VendorData]:
    """
    Loads vendor data from the database.
    """
    # Join PrebidVendorExtraction and PrebidDoc on prebid_vendor_id (implied via their own tables)
    # Actually PrebidDoc has prebid_vendor_id. PrebidVendorExtraction has prebid_vendor_id.
    # We want to join them to get a consolidated view.
    # Note: A vendor might have extraction but no doc, or vice versa.
    # But usually we work on vendors that exist.

    stmt = select(
        PrebidVendorExtraction,
        PrebidDoc
    ).join(
        PrebidDoc,
        PrebidVendorExtraction.prebid_vendor_id == PrebidDoc.prebid_vendor_id,
        isouter=True
    )

    results = session.execute(stmt).all()

    vendors = []
    for extraction, doc in results:
        # Extract URLs
        urls = set()
        if extraction.extracted_data and "domains" in extraction.extracted_data:
            domains = extraction.extracted_data["domains"]
            if isinstance(domains, list):
                for d in domains:
                    if isinstance(d, dict) and "url" in d:
                        u = d["url"]
                        if u and (u.startswith("http://") or u.startswith("https://")):
                            urls.add(u)

        doc_gpp = None
        if doc and doc.extracted_metadata:
             doc_gpp = doc.extracted_metadata.get("gpp_supported")

        vendors.append(VendorData(
            vendor_name=extraction.vendor_name,
            doc_gpp_supported=doc_gpp,
            extraction_gpp_supported=extraction.gpp_supported,
            urls=list(urls)
        ))

    return vendors

async def process_vendors(
    vendors: List[VendorData],
    concurrency: int = 10
) -> List[Dict[str, Any]]:
    """
    Async process all vendors.
    """
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = []
        # We flat map to (vendor, url) tasks or handle per vendor?
        # Handling per vendor is cleaner for organization.

        all_results = []

        # Create a coroutine for each vendor
        async def process_single_vendor(v):
            vendor_res = {
                "vendor_name": v.vendor_name,
                "doc_gpp_supported": v.doc_gpp_supported,
                "extraction_gpp_supported": v.extraction_gpp_supported,
                "endpoints": []
            }

            for url in v.urls:
                res = await check_url(session, url, sem)
                vendor_res["endpoints"].append({
                    "url": url,
                    "results": res
                })
            return vendor_res

        tasks = [process_single_vendor(v) for v in vendors]
        all_results = await asyncio.gather(*tasks)
        return all_results

def calculate_stats(results: List[Dict[str, Any]]):
    """
    Calculates and prints statistics.
    """

    # Categories of vendors
    # 1. All prebid vendors
    # 2. Docs != False (includes "check with bidder", None, True) -> basically not explicitly False
    # 3. Docs == True
    # 4. Extraction == True
    # 5. Docs == True AND Extraction == True

    categories = {
        "1. All vendors": lambda v: True,
        "2. Docs != False": lambda v: v["doc_gpp_supported"] is not False,
        "3. Docs == True": lambda v: v["doc_gpp_supported"] is True,
        "4. Extraction == True": lambda v: v["extraction_gpp_supported"] is True,
        "5. Docs == True AND Extraction == True": lambda v: (v["doc_gpp_supported"] is True) and (v["extraction_gpp_supported"] is True)
    }

    print("\n--- Prebid GPP Support Statistics ---")

    for cat_name, cat_filter in categories.items():
        # Filter vendors
        cat_vendors = [v for v in results if cat_filter(v)]
        if not cat_vendors:
            print(f"\n{cat_name}: No vendors found.")
            continue

        print(f"\n{cat_name} (n={len(cat_vendors)}):")

        # Conditions:
        # A. All endpoints / At least one endpoint
        # B. All flags / At least one flag

        # stats counters
        all_endpoints_all_flags = 0
        all_endpoints_one_flag = 0
        one_endpoint_all_flags = 0
        one_endpoint_one_flag = 0

        valid_vendors_count = 0 # Vendors with at least one URL

        for v in cat_vendors:
            endpoints = v["endpoints"]
            if not endpoints:
                continue

            valid_vendors_count += 1

            # Per endpoint analysis
            # For each endpoint, does it support all flags? Does it support at least one?
            ep_stats = []
            for ep in endpoints:
                res = ep["results"]
                # success is boolean
                flags_supported = [r["success"] for r in res.values()]

                supports_all_flags = all(flags_supported) and len(flags_supported) > 0
                supports_one_flag = any(flags_supported)

                ep_stats.append({
                    "all_flags": supports_all_flags,
                    "one_flag": supports_one_flag
                })

            # Vendor level logic
            # A. All endpoints support ...
            vendor_all_eps_all_flags = all(e["all_flags"] for e in ep_stats)
            vendor_all_eps_one_flag = all(e["one_flag"] for e in ep_stats)

            # B. At least one endpoint supports ...
            vendor_one_ep_all_flags = any(e["all_flags"] for e in ep_stats)
            vendor_one_ep_one_flag = any(e["one_flag"] for e in ep_stats)

            if vendor_all_eps_all_flags: all_endpoints_all_flags += 1
            if vendor_all_eps_one_flag: all_endpoints_one_flag += 1
            if vendor_one_ep_all_flags: one_endpoint_all_flags += 1
            if vendor_one_ep_one_flag: one_endpoint_one_flag += 1

        # Print matrix
        # Columns: All Flags | One Flag
        # Rows: All Endpoints | One Endpoint

        def fmt_pct(val, total):
            if total == 0: return f"{val} (0.0%)"
            return f"{val} ({val/total*100:.1f}%)"

        print(f"  Vendors with URLs: {valid_vendors_count}")
        print(f"  {'':<20} | {'All Regions':<15} | {'One Region':<15}")
        print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*15}")
        print(f"  {'All Endpoints':<20} | {fmt_pct(all_endpoints_all_flags, len(cat_vendors)):<20} | {fmt_pct(all_endpoints_one_flag, len(cat_vendors)):<20}")
        print(f"  {'One Endpoint':<20} | {fmt_pct(one_endpoint_all_flags, len(cat_vendors)):<20} | {fmt_pct(one_endpoint_one_flag, len(cat_vendors)):<20}")

def main():
    parser = argparse.ArgumentParser(description="Prebid GPP Support Analysis")
    parser.add_argument("--input-json", help="Path to existing JSON results to analyze (skips requests)")
    parser.add_argument("--output-json", default="data/prebid_gpp_support_results.json", help="Path to save JSON results")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")

    args = parser.parse_args()

    results = []

    if args.input_json:
        print(f"Loading results from {args.input_json}...")
        try:
            with open(args.input_json, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            sys.exit(1)
    else:
        # fetch from DB
        print("Loading vendors from database...")
        config = load_app_config()
        engine = get_engine(config)
        session_factory = get_session_factory(engine)

        with session_factory() as session:
            vendors = get_vendor_data(session)

        print(f"Found {len(vendors)} vendors in database.")

        # Run requests
        print(f"Starting async requests with concurrency {args.concurrency}...")
        try:
            results = asyncio.run(process_vendors(vendors, args.concurrency))
        except KeyboardInterrupt:
            print("Interrupted.")
            sys.exit(1)


        # Save results
        print(f"Saving results to {args.output_json}...")
        try:
             with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON: {e}")

    # Deduplicate results by vendor_name (handle potential duplicates from joins)
    unique_results = {}
    for r in results:
        v_name = r["vendor_name"]
        if v_name not in unique_results:
            unique_results[v_name] = r
        else:
            # Merge or keep first?
            # If doc_gpp_supported is False/None in one and True in another, we might want True?
            # For now, keeping first is simple enough given low volume of dupes (6).
            pass

    final_results = list(unique_results.values())
    if len(final_results) < len(results):
        print(f"Deduplicated {len(results)} -> {len(final_results)} vendors (removed {len(results) - len(final_results)} duplicates).")

    # Calculate stats
    calculate_stats(final_results)

if __name__ == "__main__":
    main()
