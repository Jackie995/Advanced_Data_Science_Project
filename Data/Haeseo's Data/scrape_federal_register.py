"""
Made entirely by Claude. - Danny

Federal Register Immigration Policy Scraper
============================================
Scrapes full-text immigration policy documents from the Federal Register API
for years 2001–2023, covering DHS agencies (ICE, CBP, USCIS).

Agency slug notes
-----------------
* ICE, CBP, and USCIS were created on March 1, 2003 when DHS absorbed the INS.
  The Federal Register API requires the full "u-s-" prefix in their slugs.
* For 2001–2002 we query the predecessor agency: immigration-and-naturalization-service.

Output: policy_text_2001_2023.csv

Requirements:
    pip install requests pandas beautifulsoup4
"""

import time
import re
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://www.federalregister.gov/api/v1/documents.json"
START_YEAR = 2003
END_YEAR = 2023
OUTPUT_FILE = "policy_text_2003_2023.csv"

# Correct slugs as registered in the Federal Register API (note the "u-s-" prefix)
AGENCY_SLUGS = [
    "u-s-immigration-and-customs-enforcement",   # ICE
    "u-s-customs-and-border-protection",         # CBP
    "u-s-citizenship-and-immigration-services",  # USCIS
]

DOC_TYPES = ["RULE", "PRORULE", "PRESDOCU"]

# Federal Register API supports up to 1000 results per page
RESULTS_PER_PAGE = 1000

# Polite delay between raw-text fetches (seconds)
RAW_TEXT_DELAY = 0.15

# Retry settings for transient network errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds; doubles on each retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_with_retry(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """GET request with exponential-backoff retry on transient errors."""
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay))
                log.warning("Rate-limited (429). Waiting %s s ...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            log.warning("Request error (%s). Retry %d/%d in %.1f s ...", exc, attempt, MAX_RETRIES, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed to GET {url} after {MAX_RETRIES} attempts")


def clean_text(raw: str) -> str:
    """
    Strip HTML tags, collapse whitespace, and drop non-ASCII characters
    so the resulting text is suitable for NLP pipelines and CSV storage.
    """
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[\r\n\t\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_agency_name(agencies: list[dict]) -> str:
    """
    Return the most specific sub-agency name found in the agencies list,
    falling back to the first agency's name if none match.
    """
    priority_keywords = ["immigration", "customs", "citizenship", "border", "naturalization"]
    names = [a.get("name", "") for a in agencies]

    for name in names:
        if any(kw in name.lower() for kw in priority_keywords):
            return name

    return names[0] if names else "Unknown"


def fetch_raw_text(raw_text_url: str) -> str:
    """Download and clean the full body text of a Federal Register document."""
    if not raw_text_url:
        return ""
    try:
        resp = _get_with_retry(raw_text_url)
        return clean_text(resp.text)
    except Exception as exc:
        log.error("Could not fetch raw text from %s: %s", raw_text_url, exc)
        return ""


# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------

def fetch_documents_for_agency_year(agency_slug: str, year: int) -> list[dict]:
    """
    Page through the Federal Register API for one agency slug + year combination
    and return raw document metadata dicts.
    """
    docs: list[dict] = []
    page = 1

    while True:
        params = {
            "conditions[agencies][]": agency_slug,
            "conditions[publication_date][year]": year,
            "conditions[type][]": DOC_TYPES,
            "fields[]": [
                "document_number",
                "title",
                "publication_date",
                "agencies",
                "raw_text_url",
                "type",
            ],
            "per_page": RESULTS_PER_PAGE,
            "page": page,
            "order": "oldest",
        }

        try:
            resp = _get_with_retry(API_BASE, params=params)
            data = resp.json()
        except Exception as exc:
            log.error("API error for agency=%s year=%d page=%d: %s", agency_slug, year, page, exc)
            break

        results: list[dict] = data.get("results", [])
        total_pages: int = data.get("total_pages", 1)
        total_count: int = data.get("count", 0)

        log.info(
            "    [%s] Page %d/%d -- %d docs total",
            agency_slug, page, total_pages, total_count,
        )
        docs.extend(results)

        if page >= total_pages:
            break
        page += 1

        time.sleep(0.05)

    return docs


def fetch_documents_for_year(year: int) -> list[dict]:
    """
    Query ICE, CBP, and USCIS for the given year, merge results, and
    deduplicate by document_number so joint publications are only processed once.
    """
    log.info("-- Year %d -----------------------------------------", year)
    seen: set[str] = set()
    all_docs: list[dict] = []

    for slug in AGENCY_SLUGS:
        docs = fetch_documents_for_agency_year(slug, year)
        for doc in docs:
            doc_num = doc.get("document_number", "")
            if doc_num and doc_num not in seen:
                seen.add(doc_num)
                all_docs.append(doc)

    log.info("  %d unique documents across ICE / CBP / USCIS for %d", len(all_docs), year)
    return all_docs


def process_documents(docs: list[dict], year: int) -> list[dict]:
    """
    For each metadata record, fetch the full body text and assemble the
    row dict matching the required CSV schema.
    """
    rows: list[dict] = []
    total = len(docs)

    for idx, doc in enumerate(docs, start=1):
        doc_number = doc.get("document_number", "")
        title = doc.get("title", "")
        pub_date = doc.get("publication_date", "")
        agencies = doc.get("agencies", [])
        raw_text_url = doc.get("raw_text_url", "")

        department = extract_agency_name(agencies)

        log.info("  [%d/%d] %s -- %s", idx, total, doc_number, title[:70])

        article = fetch_raw_text(raw_text_url)
        time.sleep(RAW_TEXT_DELAY)

        rows.append(
            {
                "Year": year,
                "Date": pub_date,
                "Department": department,
                "Title": title,
                "Article": article,
                "Document_Number": doc_number,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []

    for year in range(START_YEAR, END_YEAR + 1):
        docs = fetch_documents_for_year(year)

        if not docs:
            log.info("  No documents found for %d -- skipping.", year)
            continue

        rows = process_documents(docs, year)
        all_rows.extend(rows)

        log.info("  + %d rows collected for %d (running total: %d)", len(rows), year, len(all_rows))

        # Incremental save after each year so progress is never lost
        _save_csv(all_rows, OUTPUT_FILE)
        log.info("  Saved -> %s", OUTPUT_FILE)

    log.info("=" * 60)
    log.info("Done. Total rows: %d. Final output: %s", len(all_rows), OUTPUT_FILE)


def _save_csv(rows: list[dict], path: str) -> None:
    df = pd.DataFrame(rows, columns=["Year", "Date", "Department", "Title", "Article", "Document_Number"])
    df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
