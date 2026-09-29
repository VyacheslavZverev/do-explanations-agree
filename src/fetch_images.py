"""Build the candidate image list from Wikimedia Commons. Downloads no images.

Commons search ranking changes over time, so re-running the query is not
guaranteed to return the same files. `data/sources.csv`, written here, is what
makes the study reproducible: it pins the exact files, licences and URLs used.
That file is committed to the repository; the images themselves are not.

To reproduce the study, do NOT run this script - use the committed
`data/sources.csv` and start from `download_images.py`. Running this rebuilds
the list from today's search results, which will select different images. It
therefore refuses to overwrite an existing list unless called with --force.
"""

import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = ("do-explanations-agree/1.0 "
              "(https://github.com/VyacheslavZverev/do-explanations-agree)")
PER_TERM = 5
POLITE_DELAY = 2.0   # seconds between API calls; Commons returns HTTP 429 without it
MAX_RETRIES = 4
TERMS_FILE = Path("data/search_terms.csv")
OUT_FILE = Path("data/sources.csv")

# Licences that permit reuse with attribution. An image under anything else is
# dropped, however well it fits the class.
ALLOWED = ("cc0", "cc by", "cc by-sa", "public domain", "pdm")

COLUMNS = ["image_id", "group", "search_term", "expected_class",
           "commons_title", "licence", "artist", "credit_url", "download_url"]


def _plain(raw: str) -> str:
    """Commons returns metadata wrapped in HTML; keep the text only."""
    return html.unescape(re.sub("<[^>]+>", "", raw or "")).strip()


def _fetch_with_retry(request: urllib.request.Request) -> dict:
    """Call the API, backing off when Commons rate-limits us (HTTP 429)."""
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            wait = POLITE_DELAY * 2 ** (attempt + 1)
            print(f"  rate-limited, waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def search(term: str) -> list[dict]:
    """Return Commons records for one search term, in a stable order."""
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": "6",
        "gsrlimit": str(PER_TERM * 3), "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": "800",
    })
    request = urllib.request.Request(f"{API}?{query}",
                                     headers={"User-Agent": USER_AGENT})
    pages = _fetch_with_retry(request).get("query", {}).get("pages", {})
    # Sort by title so the same query always yields the same order.
    return sorted(pages.values(), key=lambda page: page["title"])


def to_row(page: dict, row: dict, index: int) -> dict | None:
    """Turn one Commons record into a CSV row, or None if the licence is not allowed."""
    info = page["imageinfo"][0]
    licence = _plain(info["extmetadata"].get("LicenseShortName", {}).get("value", ""))
    if not licence.lower().startswith(ALLOWED):
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", row["expected_class"].lower()).strip("_")
    return {
        "image_id": f"{slug}_{index:02d}",
        "group": row["group"],
        "search_term": row["search_term"],
        "expected_class": row["expected_class"],
        "commons_title": page["title"],
        "licence": licence,
        "artist": _plain(info["extmetadata"].get("Artist", {}).get("value", "")),
        "credit_url": info["descriptionurl"],
        "download_url": info.get("thumburl") or info["url"],
    }


def main() -> None:
    if OUT_FILE.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT_FILE} already exists and pins the images this study used.\n"
                 "Re-running the search would select different ones. To reproduce the\n"
                 "study, skip this step and run download_images.py. To build a new\n"
                 "candidate list anyway, pass --force.")
    terms = list(csv.DictReader(TERMS_FILE.open(encoding="utf-8")))
    rows, skipped = [], 0
    for row in terms:
        kept = 0
        for page in search(row["search_term"]):
            if kept == PER_TERM:
                break
            candidate = to_row(page, row, kept + 1)
            if candidate is None:
                skipped += 1
                continue
            rows.append(candidate)
            kept += 1
        print(f"{row['search_term']:20s} kept {kept}")
        time.sleep(POLITE_DELAY)

    with OUT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} candidates -> {OUT_FILE}  ({skipped} dropped on licence)")


if __name__ == "__main__":
    main()
