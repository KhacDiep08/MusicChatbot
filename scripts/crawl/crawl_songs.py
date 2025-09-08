import requests
from bs4 import BeautifulSoup
import json
import time
import random
import re

API_KEY = "10e559d2d3fd8e8c0f35f4f00fdec679"  # điền API key Last.fm của bạn

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://www.google.com/",
    "Accept-Language": "en-US,en;q=0.9"
}

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', '+', text.strip())

def crawl_azlyrics(url: str) -> dict:
    """Crawl lyrics + title + artist từ AZLyrics, xử lý nếu bị chặn"""
    response = requests.get(url, headers=HEADERS)
    if "request for access" in response.text.lower():
        print(f"[WARN] AZLyrics chặn truy cập: {url}")
        return {"title": None, "artist": None, "lyrics": None}

    soup = BeautifulSoup(response.text, "html.parser")
    raw_title = soup.title.text.strip() if soup.title else ""
    title, artist = None, None
    if "-" in raw_title:
        artist, song = raw_title.split("-", 1)
        artist = artist.strip()
        title = song.replace("Lyrics", "").replace("| AZLyrics.com", "").strip()

    lyrics = None
    divs = soup.find_all("div", class_=None, id=None)
    for div in divs:
        text = div.get_text("\n", strip=True)
        if len(text.split()) > 30:
            lyrics = text
            break

    return {"title": title, "artist": artist, "lyrics": lyrics}

def crawl_lastfm(title: str, artist: str) -> dict:
    """Crawl genre + tags từ Last.fm API"""
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "track.getInfo",
            "api_key": API_KEY,
            "artist": artist,
            "track": title,
            "format": "json"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if "track" not in data:
            print(f"[Last.fm] Không tìm thấy: {title} - {artist}")
            return {"genre": None, "tags": []}
        tags = [t["name"] for t in data["track"].get("toptags", {}).get("tag", [])]
        return {"genre": tags[0] if tags else None, "tags": tags}
    except Exception as e:
        print(f"[Last.fm ERROR] {title} - {artist} => {e}")
        return {"genre": None, "tags": []}

def crawl_song(azlyrics_url: str) -> dict:
    az_data = crawl_azlyrics(azlyrics_url)
    if not az_data["title"] or not az_data["artist"]:
        print(f"[WARN] Không tách được title/artist từ {azlyrics_url}")
        return {}
    lastfm_data = crawl_lastfm(az_data["title"], az_data["artist"])
    return {**az_data, **lastfm_data}

def crawl_from_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    all_data = []
    for idx, url in enumerate(urls):
        print(f"[{idx+1}/{len(urls)}] Đang xử lý: {url}")
        try:
            data = crawl_song(url)
            if data:
                all_data.append(data)
        except Exception as e:
            print(f"[ERROR] {url} => {e}")
        time.sleep(random.uniform(1, 3))  # delay ngẫu nhiên tránh bị chặn

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Crawl hoàn tất! Đã lưu vào '{output_file}'")
