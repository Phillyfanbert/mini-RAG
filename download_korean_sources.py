from pathlib import Path
import csv
import json
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGDownloader/1.0)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

SOURCES = [
    {
        "name": "Lets_Chat_Korean",
        "kind": "pressbooks",
        "page_url": "https://boisestate.pressbooks.pub/pathwayskorean/",
        "downloads": [
            "https://boisestate.pressbooks.pub/pathwayskorean/open/download?type=pdf",
            "https://boisestate.pressbooks.pub/pathwayskorean/open/download?type=epub",
        ],
    },
    {
        "name": "Tammy_Korean_PDFs",
        "kind": "tammy_pdfs",
        "page_url": "https://learning-korean.com/pdf/",
    },
    {
        "name": "Tammy_Korean_Lessons",
        "kind": "tammy_lessons",
        "archive_url": "https://learning-korean.com/elementary/",
    },
    {
        "name": "Beginning_Korean_1",
        "kind": "open_textbook_page",
        "page_url": "https://open.umn.edu/opentextbooks/textbooks/beginning-korean-1",
        "notes": "Open textbook landing page; may require manual file discovery.",
    },
    {
        "name": "My_Korean_1_2",
        "kind": "talkingtokoreans",
        "page_url": "https://talkingtokoreans.com/my-korean",
    },
    {
        "name": "Basic_English_for_Teaching_Korean",
        "kind": "orda",
        "page_url": "https://orda.shef.ac.uk/articles/book/_b_Basic_English_for_Teaching_Korean_b_b____b_/24103806",
    },
]

BASE = "https://learning-korean.com"
ARCHIVE_START = "https://learning-korean.com/elementary/"

def safe_name(s):
    s = re.sub(r"[^\w\-\.]+", "_", s.strip())
    return s[:180]

def fetch(url):
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    return r

def download(url, out_path):
    r = fetch(url)
    out_path.write_bytes(r.content)
    return out_path

def discover_pdf_links(page_url):
    html = fetch(page_url).text
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else urljoin(page_url, href)
        if full.lower().endswith(".pdf"):
            links.append(full)
    return list(dict.fromkeys(links))

def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    selectors = ["article", "main", ".entry-content", ".post-content", ".content-area"]
    best = ""
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text("\n", strip=True)
            if len(txt) > len(best):
                best = txt
    return best if best else soup.get_text("\n", strip=True)

def get_archive_page_urls(start_url=ARCHIVE_START, max_pages=50):
    seen = set()
    queue = [start_url]
    pages = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        pages.append(url)

        try:
            html = fetch(url).text
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0].rstrip("/")
            path = urlparse(href).path.rstrip("/")
            if "/elementary/page/" in path and href not in seen:
                if href not in queue and href not in pages:
                    queue.append(href)

    return pages

def collect_lesson_urls():
    urls = set()
    archive_urls = get_archive_page_urls()

    for page in archive_urls:
        try:
            html = fetch(page).text
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(page, a["href"]).split("#")[0].rstrip("/")
            path = urlparse(href).path.rstrip("/")
            if re.match(r"^/elementary/\d{8}-\d+/?$", path):
                urls.add(href)

    return sorted(urls)

def collect_pressbooks():
    manifest = []
    for src in SOURCES:
        if src["kind"] != "pressbooks":
            continue
        print(f"\n== {src['name']} ==")
        for i, url in enumerate(src["downloads"], 1):
            ext = Path(url.split("?")[0]).suffix.lower()
            if ext not in [".pdf", ".epub"]:
                continue
            out_path = OUT_DIR / safe_name(f"{src['name']}_{i}{ext}")
            try:
                download(url, out_path)
                print(f"Downloaded: {url} -> {out_path}")
                manifest.append({
                    "type": "pressbooks",
                    "title": out_path.stem,
                    "url": url,
                    "file": str(out_path),
                })
            except Exception as e:
                print(f"Failed: {url} ({e})")
    return manifest

def collect_tammy_pdfs():
    manifest = []
    print("\n== Tammy_Korean_PDFs ==")
    try:
        pdf_urls = discover_pdf_links("https://learning-korean.com/pdf/")
    except Exception as e:
        print(f"Failed to fetch Tammy PDF page: {e}")
        return manifest

    for i, url in enumerate(pdf_urls, 1):
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext != ".pdf":
            continue
        out_path = OUT_DIR / safe_name(f"Tammy_PDF_{i}.pdf")
        try:
            download(url, out_path)
            print(f"Downloaded PDF: {url} -> {out_path}")
            manifest.append({
                "type": "pdf",
                "title": out_path.stem,
                "url": url,
                "file": str(out_path),
            })
        except Exception as e:
            print(f"Failed PDF {url}: {e}")
    return manifest

def collect_tammy_lessons():
    manifest = []
    print("\n== Tammy_Korean_Lessons ==")
    lesson_urls = collect_lesson_urls()
    print(f"Found {len(lesson_urls)} Tammy lesson pages")
    for i, url in enumerate(lesson_urls, 1):
        try:
            html = fetch(url).text
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(" ", strip=True)
            text = extract_main_text(html)
            fname = safe_name(f"{i:03d}_{title}.txt")
            out_path = OUT_DIR / fname
            out_path.write_text(f"URL: {url}\nTITLE: {title}\n\n{text}\n", encoding="utf-8")
            manifest.append({
                "type": "lesson",
                "title": title,
                "url": url,
                "file": str(out_path),
            })
            print(f"Saved lesson: {out_path}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Failed lesson {url}: {e}")
    return manifest

def collect_open_textbook_page(page_url, out_prefix):
    manifest = []
    print(f"\n== {out_prefix} ==")
    try:
        html = fetch(page_url).text
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = href if href.startswith("http") else urljoin(page_url, href)
            low = full.lower()
            text = a.get_text(" ", strip=True).lower()
            if any(low.endswith(ext) for ext in [".pdf", ".zip", ".epub", ".mp3"]) or "download" in low or "download" in text:
                links.append(full)
        links = list(dict.fromkeys(links))
        if not links:
            print(f"No direct downloads found on {page_url}")
            return manifest

        for i, url in enumerate(links, 1):
            clean = url.split("?")[0]
            ext = Path(clean).suffix.lower()
            if ext not in [".pdf", ".zip", ".epub", ".mp3"]:
                if "pdf" in clean.lower():
                    ext = ".pdf"
                elif "zip" in clean.lower():
                    ext = ".zip"
                elif "mp3" in clean.lower():
                    ext = ".mp3"
                else:
                    continue

            out_path = OUT_DIR / safe_name(f"{out_prefix}_{i}{ext}")
            try:
                download(url, out_path)
                print(f"Downloaded: {url} -> {out_path}")
                manifest.append({
                    "type": "open_textbook",
                    "title": out_path.stem,
                    "url": url,
                    "file": str(out_path),
                })
            except Exception as e:
                print(f"Failed: {url} ({e})")
    except Exception as e:
        print(f"Failed to fetch {page_url}: {e}")
    return manifest

def collect_orda(page_url, out_prefix):
    manifest = []
    print(f"\n== {out_prefix} ==")
    try:
        html = fetch(page_url).text
    except Exception as e:
        print(f"Failed to fetch ORDA page: {e}")
        return manifest

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else urljoin(page_url, href)
        low = full.lower()
        text = a.get_text(" ", strip=True).lower()
        if "download file" in text or any(low.endswith(ext) for ext in [".pdf", ".mp3", ".zip"]):
            links.append(full)

    links = list(dict.fromkeys(links))

    for i, url in enumerate(links, 1):
        clean = url.split("?")[0]
        ext = Path(clean).suffix.lower()
        if ext not in [".pdf", ".mp3", ".zip"]:
            if "pdf" in clean.lower():
                ext = ".pdf"
            elif "mp3" in clean.lower():
                ext = ".mp3"
            else:
                continue

        out_path = OUT_DIR / safe_name(f"{out_prefix}_{i}{ext}")
        try:
            download(url, out_path)
            print(f"Downloaded: {url} -> {out_path}")
            manifest.append({
                "type": "orda",
                "title": out_path.stem,
                "url": url,
                "file": str(out_path),
            })
        except Exception as e:
            print(f"Failed: {url} ({e})")
    return manifest

def collect_talkingtokoreans():
    manifest = []
    print("\n== My_Korean_1_2 ==")
    page_url = "https://talkingtokoreans.com/my-korean"
    try:
        html = fetch(page_url).text
    except Exception as e:
        print(f"Failed to fetch Talking2Koreans page: {e}")
        return manifest

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else urljoin(page_url, href)
        low = full.lower()
        if any(x in low for x in ["my-korean-1", "my-korean-2", "download"]) and any(low.endswith(ext) for ext in [".pdf", ".zip"]):
            links.append(full)
        elif "vocab" in low and low.endswith(".pdf"):
            links.append(full)
        elif "audio" in low and low.endswith(".zip"):
            links.append(full)
    links = list(dict.fromkeys(links))

    for i, url in enumerate(links, 1):
        clean = url.split("?")[0]
        ext = Path(clean).suffix.lower()
        if ext not in [".pdf", ".zip"]:
            continue
        out_path = OUT_DIR / safe_name(f"My_Korean_{i}{ext}")
        try:
            download(url, out_path)
            print(f"Downloaded: {url} -> {out_path}")
            manifest.append({
                "type": "pdf_or_zip",
                "title": out_path.stem,
                "url": url,
                "file": str(out_path),
            })
        except Exception as e:
            print(f"Failed: {url} ({e})")
    return manifest

def main():
    manifest = []
    manifest.extend(collect_pressbooks())
    manifest.extend(collect_tammy_pdfs())
    manifest.extend(collect_tammy_lessons())
    manifest.extend(collect_open_textbook_page(
        "https://open.umn.edu/opentextbooks/textbooks/beginning-korean-1",
        "Beginning_Korean_1"
    ))
    manifest.extend(collect_talkingtokoreans())
    manifest.extend(collect_orda(
        "https://orda.shef.ac.uk/articles/book/_b_Basic_English_for_Teaching_Korean_b_b____b_/24103806",
        "Basic_English_for_Teaching_Korean"
    ))

    (OUT_DIR / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    with open(OUT_DIR / "sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "title", "url", "file"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Done. Collected {len(manifest)} files into {OUT_DIR}/")

if __name__ == "__main__":
    main()