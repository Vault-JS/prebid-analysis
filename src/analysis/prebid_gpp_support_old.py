"""
Legacy Prebid.js vendor URL extractor (deprecated).

Old script for extracting vendor URLs and privacy flags from Prebid.js module files.
Replaced by prebid_loader.py which provides more comprehensive extraction.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
#!/usr/bin/env python3
from collections import defaultdict
import json
import os
import re

import requests

# Path to your local Prebid.js clone
PREBID_DIR = "Prebid.js/modules"

# Regex patterns
url_pattern = re.compile(r"https?://[^\s\"']+")
privacy_pattern = re.compile(r"\b(gppConsent|us_privacy|tcf|consent)\b", re.IGNORECASE)
gpp_pattern = re.compile(r"\b(gppConsent)\b", re.IGNORECASE)

vendors = defaultdict(
    lambda: {"urls": set(), "privacy_flags": set(), "gpp_flags": set()}
)

for root, _, files in os.walk(PREBID_DIR):
    for f in files:
        if not f.endswith(".js"):
            continue
        path = os.path.join(root, f)
        vendor = os.path.splitext(f)[0]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
                # Find URLs
                urls = url_pattern.findall(text)
                if urls:
                    vendors[vendor]["urls"].update(urls)
                # Find privacy-related flags
                priv_flags = privacy_pattern.findall(text)
                if priv_flags:
                    vendors[vendor]["privacy_flags"].update(map(str.lower, priv_flags))
                # Find GPP-related flags
                gpp_flags = gpp_pattern.findall(text)
                if gpp_flags:
                    vendors[vendor]["gpp_flags"].update(map(str.lower, gpp_flags))
        except (IOError, OSError, UnicodeDecodeError) as e:
            print(f"Error reading {path}: {e}")

# stats
print(f"Total vendors: {len(vendors)}")
print(f"Total URLs: {sum(len(v['urls']) for v in vendors.values())}")
print(f"Total privacy flags: {sum(len(v['privacy_flags']) for v in vendors.values())}")
print(f"Total GPP flags: {sum(len(v['gpp_flags']) for v in vendors.values())}")

# Prepare results
summary = []
for vendor, data in vendors.items():
    summary.append(
        {
            "vendor": vendor,
            "endpoint_count": len(data["urls"]),
            "privacy_flags": sorted(data["privacy_flags"]),
            "gpp_flags": sorted(data["gpp_flags"]),
            "example_urls": list(sorted(data["urls"]))[:5],
        }
    )

GPP_STRINGS = {
    "eu": "DBABMA~CPXxRfAPXxRfAAfKABENB-CgAAAAAAAAAAYgAAAAAAAA",
    "us-california": "DBABBg~BUoAAABg.Q",
    "us-virginia": "DBABNA~B-9Q",
    "us": "DBABLA~B-7Q",
}

gpp_suported_vendors_count = 0
# Note: This block will test each endpoint (URL) for each vendor by appending ?gpp=<string>
gpp_test_results = {}

for vendor in summary:
    urls = vendor.get("example_urls", [])
    test_results = []
    for url in urls:
        # Try for each GPP string
        for region, gpp_str in GPP_STRINGS.items():
            # Ensure we append gpp param properly
            if "?" in url:
                test_url = f"{url}&gpp={gpp_str}"
            else:
                test_url = f"{url}?gpp={gpp_str}"
            try:
                response = requests.get(test_url, timeout=5)
                status = response.status_code
                HTTP_SUCCESS_MIN = 200
                HTTP_SUCCESS_MAX = 400
                success = HTTP_SUCCESS_MIN <= status < HTTP_SUCCESS_MAX
                if success and vendor["gpp_flags"]:
                    gpp_suported_vendors_count += 1
            except requests.RequestException:
                status = None
                success = False
            test_results.append(
                {
                    "endpoint": url,
                    "gpp_region": region,
                    "gpp_value": gpp_str,
                    "test_url": test_url,
                    "success": success,
                    "status_code": status,
                }
            )
    gpp_test_results[vendor["vendor"]] = test_results

print(f"GPP suported vendors count: {gpp_suported_vendors_count}")

# If desired, output to diagnostics file for review
with open("data/prebid_gpp_endpoint_test_results.json", "w", encoding="utf-8") as out:
    json.dump(gpp_test_results, out, indent=2)


# Output summary JSON
output_path = "data/prebid_endpoints_summary.json"
with open(output_path, "w", encoding="utf-8") as out:
    json.dump(summary, out, indent=2)
