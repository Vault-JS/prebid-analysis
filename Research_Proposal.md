# Research Proposal

**Title:** The GPP Gap: Measuring the Discrepancy Between Privacy Signaling and AdTech Enforcement in the Prebid Ecosystem

**Target Venue:** PETS (PoPETs) / ACM IMC

## 1. Abstract
The Global Privacy Platform (GPP) string is the emerging standard for unifying privacy preferences across jurisdictions (GDPR TCF, US MSPA, etc.). While adoption is growing, enforcement mechanisms in the server-side AdTech ecosystem remain opaque. This study proposes a comprehensive audit of Prebid.js vendors to determine if claimed GPP support translates to actual privacy compliance. Using a combination of static code analysis and differential dynamic testing, we aim to measure the gap between *signaling availability* (accepting the string) and *enforcement* (suppressing data collection and behavioral bidding).

## 2. Motivation & Problem Statement
Prebid.js is the dominant open-source header bidding wrapper. Prebid.js supports vendor declaration of GPP adoption to manage regulatory compliance. However, the architecture of Real-Time Bidding (RTB) relies heavily on trust; once a bid request leaves the browser, the user loses control over their data.

**Preliminary findings** suggest that "support" is often superficial:
*   **55/577** vendors claim GPP support in documentation.
  * 94.4% of them accept GPP payloads (return 200-399 status).
*   **89/577** vendors implement GPP logic in their source code (detected via LLM static analysis).
  * 100% of them accept GPP payloads.

This shows significant mismatch between claimed support and actual implementation.

However, an HTTP 200 OK response indicates only that the server *parsed* the request, not that it *honored* the privacy signal. There is currently no large-scale empirical evidence quantifying how many of these vendors actively suppress tracking pixels, cookies, or behavioral bidding logic when receiving a valid GPP Opt-Out string.

## 3. Research Questions (RQs)
*   **RQ1 (implementation):** To what extent does the Prebid.js open-source codebase reflect actual server-side capability for GPP processing?
*   **RQ2 (Technical Compliance):** Do vendors persist identifiers (cookies) or initiate third-party user-syncing flows (pixels) in response to requests containing GPP Opt-Out signals?
*   **RQ3 (Economic Compliance):** Can we detect behavioral bidding (price discrimination) on users who have opted out, indicating illegal processing of personal data for ad targeting?

## 4. Methodology

The study will proceed in three phases:

### Phase I: Static Analysis & Endpoint Extraction (Completed/Refining)
We utilize the Prebid.js GitHub repository to extract:
1.  **Vendor List:** All adapters claiming GPP support.
2.  **Endpoints:** The specific URLs used for bid requests.
3.  **Client-Side Logic:** Using an LLM, we categorize adapters based on whether they block requests client-side (in the browser) or forward flags to the server. *This establishes the ground truth for who "should" be compliant.*

### Phase II: The "Cookie Trap" (Technical Compliance - RQ2)
This test detects objective violations of storage consent (e.g., "No Sale/Sharing" or TCF Purpose 1 revocation).

*   **Procedure:** Send synthetic OpenRTB requests to extracted endpoints.
*   **Variables:**
    *   **Control:** GPP String = `Null` or Full Consent.
    *   **Test:** GPP String = Strict Opt-Out (e.g., MSPA `OptOutSale`, TCF `No Consent`).
*   **Measurement:** Inspect HTTP Response Headers and Body.
    *   **Violation A:** Server returns `Set-Cookie` header on Test group.
    *   **Violation B:** Server returns HTML/JSON triggering a "User Sync" (pixel `<img>` or `<iframe>`) to 3rd party domains.
*   **Significance:** This provides a binary, undeniable metric of non-compliance.

### Phase III: The "Price Indifference" Study (Economic Compliance - RQ3)
This test attempts to detect if vendors are using PII (Personal Identifiable Information) internally to value the impression, even if they don't set new cookies.

*   **Persona Generation:** We will crawl the web to generate "Warm" user profiles (valid, high-value cookies/IDs for major SSPs).
*   **Triangulated Requests:** For a specific inventory unit, we issue three simultaneous requests:
    1.  **Contextual Baseline:** No User ID, GPP Opt-Out. (Price = $X)
    2.  **The Trap:** Warm User ID, GPP Opt-Out. (Price = $Y)
    3.  **Positive Control:** Warm User ID, GPP Full Consent. (Price = $Z)
*   **Analysis:**
    *   If $Y \approx X$: Compliant (Vendor ignored the ID).
    *   If $Y \approx Z$ AND $Y \gg X$: **Violation**. The vendor matched the ID and bid based on behavioral history despite the Opt-Out.

## 5. Expected Contributions
1.  **Taxonomy of GPP Support:** A categorization of how vendors implement GPP (client-side filter vs. server-side processing).
2.  **Quantification of "Fake" Support:** A specific percentage of vendors who return `200 OK` but fail the "Cookie Trap" or "Price Indifference" tests.
3.  **Open Data:** Publication of the test strings and GPP payloads to allow vendors to self-audit.

## 6. Ethical Considerations
*   **Fake Traffic:** We will use `test: 1` flags where supported. Where live bidding is necessary for Phase III, we will minimize QPS to avoid distorting campaign analytics.
*   **No Ad Interaction:** We will not render or click ads, ensuring no fraudulent costs are incurred by advertisers, only minimal compute costs for SSPs.
*   **Responsible Disclosure:** If systemic violations are found (e.g., a vendor ignoring opt-outs globally), we will attempt to contact the vendor 30 days prior to publication.

---

### Critical Advice for the "Publisher" aspect:

If you want to publish this at **PETS**:
1.  **Focus on Phase II (Cookies/Pixels).** It is the "hard science." Phase III (Pricing) is noisy and reviewers might argue about bidding logic. Phase II is binary: *Did they set a cookie when I said no? Yes/No.* That is irrefutable.
2.  **Scale.** You need to test as many vendors as possible. Testing 5 vendors is a blog post. Testing 150 vendors is a paper.
  - Yes, we have about 570 vendors in Prebid.js.
3.  **LLM Validation.** Since you used an LLM to extract the GPP logic, you must manually verify a random sample (e.g., 10%) of the code to calculate the LLM's accuracy. You cannot blindly trust the LLM in the methodology section of a paper.
  - DEfinitely planned, so far I checked 2-3 vendors and it seemed accurate, but we need a proper validation process.
