"""Step 3b - download the candidate images listed in data/sources.csv.

Safe to re-run: files already on disk are skipped, so an interrupted run
continues where it stopped. Every download is opened afterwards to confirm it
is a real image - a truncated file or an HTML error page looks perfectly
normal in a directory listing.

After downloading, each image is compared with data/image_checksums.csv, a
digest of the pixels this study actually used. Wikimedia can serve the same
URL with different bytes later: in a re-download three weeks on, 8 of 60 files
differed only in metadata (identical pixels) and one had been re-encoded. File
bytes would raise false alarms, so the digest covers decoded RGB pixels only.
`--record-checksums` writes that file from the images currently on disk.
"""

import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

USER_AGENT = "xai-agreement-student-project/0.1 (educational, non-commercial)"
POLITE_DELAY = 0.5
SOURCES = Path("data/sources.csv")
IMAGE_DIR = Path("data/images")
CHECKSUMS = Path("data/image_checksums.csv")


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


def pixel_digest(path: Path) -> str:
    """SHA-256 of an image's decoded RGB pixels and size - not of the file bytes."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        header = f"{rgb.width}x{rgb.height}:".encode()
        return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def record_checksums(rows: list[dict]) -> None:
    """Write the pixel digests of the images on disk - run once, by the author."""
    with CHECKSUMS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "pixel_sha256"])
        for row in rows:
            writer.writerow([row["image_id"], pixel_digest(target_path(row))])
    print(f"{len(rows)} pixel digests -> {CHECKSUMS}")


def verify_checksums(rows: list[dict]) -> None:
    """Report any image whose pixels differ from the copy this study used."""
    if not CHECKSUMS.exists():
        return
    with CHECKSUMS.open(encoding="utf-8") as handle:
        expected = {r["image_id"]: r["pixel_sha256"] for r in csv.DictReader(handle)}
    changed = [row["image_id"] for row in rows
               if target_path(row).exists()
               and pixel_digest(target_path(row)) != expected.get(row["image_id"])]
    if changed:
        print(f"WARNING: {len(changed)} image(s) differ in pixels from the study's copy, "
              f"so their results may differ slightly: {', '.join(changed)}")
    else:
        print("all images match the study's pixels")


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SOURCES.open(encoding="utf-8")))
    if "--record-checksums" in sys.argv:
        record_checksums(rows)
        return
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
    verify_checksums(rows)


if __name__ == "__main__":
    main()
