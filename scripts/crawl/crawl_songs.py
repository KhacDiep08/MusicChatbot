import requests
from bs4 import BeautifulSoup
import json
import time
import re


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản để đưa vào URL của Last.fm (dùng dấu + thay vì khoảng trắng)."""
    return re.sub(r'\s+', '+', text.strip())


def crawl_azlyrics(url: str) -> dict:
    """Crawl title, artist và lyrics từ trang AZLyrics."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Lấy title và artist từ <title>
        title_tag = soup.title.string.strip()
        match = re.match(r"(.*?)\s*-\s*(.*?)\s*Lyrics", title_tag)
        if not match:
            raise ValueError("Không tách được title và artist từ tiêu đề")

        artist = match.group(1).strip().lower()  # giữ chữ thường cho Last.fm
        title = match.group(2).strip().lower()

        # Lấy lyrics từ div không class
        lyrics = ""
        for div in soup.find_all("div", class_=False, id=False):
            text = div.get_text(separator="\n").strip()
            if text and len(text.split()) > 20:
                lyrics = text
                break

        return {
            "title": title,
            "artist": artist,
            "lyrics": lyrics
        }

    except Exception as e:
        print(f"[!] Lỗi khi crawl AZLyrics: {e}")
        return {
            "title": "",
            "artist": "",
            "lyrics": ""
        }


def crawl_lastfm(title: str, artist: str) -> dict:
    """Crawl genre và tags từ trang Last.fm (dùng artist/title dạng thường)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        artist_url = normalize_text(artist)
        title_url = normalize_text(title)
        url = f"https://www.last.fm/music/{artist_url}/{title_url}"

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        tags = []
        genre = ""

        tag_ul = soup.find("ul", class_="tags-list")
        if tag_ul:
            tags = [li.get_text(strip=True) for li in tag_ul.find_all("li") if li.text.strip()]

        if tags:
            genre = tags[0]

        return {
            "genre": genre,
            "tags": tags
        }

    except Exception as e:
        print(f"[!] Lỗi khi crawl Last.fm ({title} - {artist}): {e}")
        return {
            "genre": "",
            "tags": []
        }


def crawl_song(azlyrics_url: str) -> dict:
    """Kết hợp dữ liệu từ AZLyrics và Last.fm."""
    az_data = crawl_azlyrics(azlyrics_url)
    if not az_data["title"] or not az_data["artist"]:
        return {}

    lastfm_data = crawl_lastfm(az_data["title"], az_data["artist"])

    return {
        "title": az_data["title"],
        "artist": az_data["artist"],
        "lyrics": az_data["lyrics"],
        "genre": lastfm_data["genre"],
        "tags": lastfm_data["tags"]
    }


def crawl_from_file(input_file: str, output_file: str):
    """Đọc danh sách URL từ file, crawl dữ liệu và lưu kết quả."""
    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    all_data = []

    for idx, url in enumerate(urls):
        print(f"[{idx + 1}/{len(urls)}] Đang xử lý: {url}")
        data = crawl_song(url)
        if data:
            all_data.append(data)
        time.sleep(2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Crawl hoàn tất! Đã lưu vào '{output_file}'")
