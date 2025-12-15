# Prebid analysis

This repository contains scripts for analyzing Prebid.js codebase and inspecting GPP support of Prebid vendors.
It is meant as an exploration repo for https://github.com/Vault-JS/vendor-library-llm

## Setup

- Clone Prebid repositories:
  - `git clone git@github.com:prebid/Prebid.js.git`
  - `git clone git@github.com:prebid/prebid.github.io.git`
- Install dependencies using `uv`
  - `uv sync`
- Create `.env` with `OPENAI_API_KEY`

## Results


### Repo comparison: prebid.github.io Docs vs LLM Extraction from Prebid.js

```
❯ uv run python -m src.analysis.inspect_prebid
Fetching data...
Loaded 577 extractions, 717 docs, 712 CSV rows.

================================================================================
COMPARISON: PrebidDoc vs LLM Extraction
================================================================================
--- GDPR Supported ---
Count: 425
Accuracy (manual):  71.76%
F1 Score: 0.74
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   132  |  77
Doc: True    43   |  173

--- USP / CCPA Supported ---
Count: 425
Accuracy (manual):  70.82%
F1 Score: 0.67
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   173  |  24
Doc: True    100  |  128

--- COPPA Supported ---
Count: 425
Accuracy (manual):  74.12%
F1 Score: 0.50
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   260  |  15
Doc: True    95   |  55

--- Schains (Supply Chain) Supported ---
Count: 425
Accuracy (manual):  64.71%
F1 Score: 0.55
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   183  |  15
Doc: True    135  |  92

--- Floors Supported ---
Count: 425
Accuracy (manual):  48.94%
F1 Score: 0.00
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   208  |   0
Doc: True    217  |   0

--- S2S Supported (PBS) ---
Count: 425
Accuracy (manual):  52.71%
F1 Score: 0.00
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   224  |   0
Doc: True    201  |   0

--- GPP Supported ---
Count: 425
Accuracy (manual):  84.94%
F1 Score: 0.51
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   328  |  40
Doc: True    24   |  33

--- App Supported ---
Count: 425
Accuracy (manual):  79.53%
F1 Score: 0.00
Confusion Matrix:
       LLM: False | LLM: True
Doc: False   338  |   0
Doc: True    87   |   0

--- Bidder Code ---
Count: 425
Accuracy (manual):  99.53%
Disagreements: 2

--- Media Types ---
Count: 425
Accuracy (manual):  77.41%
(Complex type comparison: Exact match only)
Disagreements: 96
Sample Disagreement: Doc=() vs Extracted=('banner',)

--- User IDs ---
Count: 425
Accuracy (manual):  61.88%
(Complex type comparison: Exact match only)
Disagreements: 162
Sample Disagreement: Doc={'parrableid', 'unifiedid', 'pubcommonid', 'identitylink', 'liveintentid', 'netid', 'id5id', 'criteo', 'britepoolid'} vs Extracted=set()


================================================================================
COMPARISON: PrebidDoc vs CSV Data
================================================================================
media_types         : Accuracy: 77.42% (552/713)
  Top 5 Mismatches:
    - DAN Marketplace: Doc=set() != CSV={'banner'}
    - Supply2: Doc=set() != CSV={'banner'}
    - Go2Net: Doc={'video'} != CSV={'video', 'banner'}
    - Between: Doc=set() != CSV={'banner'}
    - Criteo: Doc={'video', 'display', 'native (pbjs only)'} != CSV={'native', 'video', 'banner'}
schain_supported    : Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - TrustX: Doc=True != CSV=False (RowVal='check with bidder')
tcfeu_supported     : Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - TrustX: Doc=True != CSV=False (RowVal='no')
usp_supported       : Accuracy: 100.00% (713/713)
coppa_supported     : Accuracy: 99.72% (711/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
    - TrustX: Doc=True != CSV=False (RowVal='check with bidder')
floors_supported    : Accuracy: 99.44% (709/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
    - TrustX: Doc=True != CSV=False (RowVal='check with bidder')
    - Outbrain: Doc=True != CSV=False (RowVal='check with bidder')
    - AMX RTB: Doc=False != CSV=True (RowVal='yes')
gpp_supported       : Accuracy: 88.08% (628/713)
  Top 5 Mismatches:
    - Criteo: Doc=True != CSV=False (RowVal='some (check with bidder)')
    - StackAdapt: Doc=True != CSV=False (RowVal='tcfeu  tcfca  usnat  usstate_all  usp')
    - Cadent Aperture MX: Doc=True != CSV=False (RowVal='some (check with bidder)')
    - Tagoras: Doc=True != CSV=False (RowVal='some (check with bidder)')
    - Boldwin: Doc=True != CSV=False (RowVal='some (check with bidder)')
dchain_supported    : Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
multiformat_supported: Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
safeframes_ok       : Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
deals_supported     : Accuracy: 99.86% (712/713)
  Top 5 Mismatches:
    - Attekmi: Doc=True != CSV=False (RowVal='check with bidder')
ortb_blocking_supported: Accuracy: 100.00% (713/713)
user_ids            : Accuracy: 98.74% (704/713)
  Top 5 Mismatches:
    - Digital Matter: Doc=none != CSV=set()
    - TrustX: Doc=all != CSV=set()
    - Outbrain: Doc=id5Id, identityLink != CSV=set()
    - Index Exchange (Prebid Server): Doc=idl, netId, fabrickId, zeotapIdPlus, uid2, TDID, id5Id, lotamePanoramaId, publinkId, hadronId, pubcid, utiq, criteoID, euid, imuid, 33acrossId, nonID, pairid != CSV={'lotamepanoramaid', 'imuid', 'trustpid', 'euid', 'utiqmtpid', 'criteoid', 'nonid', 'tpid', 'tdid', 'hadronid', 'connectid', 'pubcid', 'm1id', 'publinkid', 'fabrickid', 'pairid', 'amazonadvertisingid', '33acrossid', 'zeotapidplus', 'rampid', 'id5id'}
    - flipp: Doc=none != CSV=set()
```

#### Comments

- CSV seems to reflect documentation state close, but small changes were found - it is more reliable to extract the data from docs manually (more reliable)
- Documentation and code (LLM Extraction) show large mismatch. Based on a very limited manual inspection, it seems that LLM is reliable in property extraction, and the documentation is not reflecting the code well. Anyway, we should ingest both for vendor library, but use LLM extraction as the primary source.


### GPP support

```
❯ uv run python -m src.analysis.prebid_gpp_support # takes a few minutes
Deduplicated 583 -> 577 vendors (removed 6 duplicates).

--- Prebid GPP Support Statistics ---

1. All vendors (n=577):
  Vendors with URLs: 546
                       | All Regions     | One Region
  ---------------------+-----------------+----------------
  All Endpoints        | 126 (21.8%)     | 128 (22.2%)
  One Endpoint         | 282 (48.9%)     | 286 (49.6%)

2. Docs != False (n=567):
  Vendors with URLs: 537
                       | All Regions     | One Region
  ---------------------+-----------------+----------------
  All Endpoints        | 123 (21.7%)     | 125 (22.0%)
  One Endpoint         | 277 (48.9%)     | 281 (49.6%)

3. Docs == True (n=55):
  Vendors with URLs: 54
                       | All Regions     | One Region
  ---------------------+-----------------+----------------
  All Endpoints        | 9 (16.4%)       | 9 (16.4%)
  One Endpoint         | 34 (61.8%)      | 34 (61.8%)

4. Extraction == True (n=89):
  Vendors with URLs: 86
                       | All Regions     | One Region
  ---------------------+-----------------+----------------
  All Endpoints        | 18 (20.2%)      | 19 (21.3%)
  One Endpoint         | 62 (69.7%)      | 62 (69.7%)

5. Docs == True AND Extraction == True (n=32):
  Vendors with URLs: 32
                       | All Regions     | One Region
  ---------------------+-----------------+----------------
  All Endpoints        | 8 (25.0%)       | 8 (25.0%)
  One Endpoint         | 23 (71.9%)      | 23 (71.9%)
```

#### Comments

- Surprisingly, both documented and LLM-extracted data from code are not a reliable predictor whether GPP string request are supported or not.
