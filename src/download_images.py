"""Step 3b - download the candidate images listed in data/sources.csv.

Safe to re-run: files already on disk are skipped, so an interrupted run
continues where it stopped. Every download is opened afterwards to confirm it
is a real image - a truncated file or an HTML error page looks perfectly
normal in a directory listing.
"""

import csv
import time
import urllib.request
from pathlib import Path

from PIL import Image

USER_AGENT = "xai-agreement-student-project/0.1 (educational, non-commercial)"
POLITE_DELAY = 0.5
SOURCES = Path("data/sources.csv")
IMAGE_DIR = Path("data/images")


def target_path(row: dict) -> Path:
    """Where one row's image belongs, keeping the original file extension."""
    suffix = Path(row["download_url"].split("?")[0]).suffix.lower() or ".jpg"
    return IMAGE_DIR / f"{row['image_id']}{suffix}"


def download(url: str, path: Path) -> None:
    """Fetch one file, then verify it decodes as an image."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        path.unlink(missing_ok=True)   # never leave a broken file behind
        raise


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SOURCES.open(encoding="utf-8")))
    fetched = skipped = failed = 0

    for row in rows:
        path = target_path(row)
        if path.exists():
            skipped += 1
            continue
        try:
            download(row["download_url"], path)
            fetched += 1
        except Exception as error:
            print(f"  FAILED {row['image_id']}: {type(error).__name__}: {error}")
            failed += 1
        time.sleep(POLITE_DELAY)

    print(f"\n{len(rows)} listed | {fetched} downloaded | "
          f"{skipped} already present | {failed} failed")


if __name__ == "__main__":
    main()
