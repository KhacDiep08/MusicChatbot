import json
import random

# Từ khóa cho từng mood
mood_keywords = {
    "sadness": ["cry", "tears", "lonely", "miss", "sad", "broken", "pain", "lost"],
    "anger": ["hate", "fight", "kill", "angry", "rage", "burn", "enemy"],
    "joy": ["love", "happy", "smile", "dance", "joy", "together", "shine"],
    "fear": ["fear", "die", "dark", "scared", "alone", "nightmare", "ghost"]
}

def detect_mood(lyrics: str) -> str:
    if not lyrics:
        return random.choice(list(mood_keywords.keys()))
    text = lyrics.lower()
    for mood, keywords in mood_keywords.items():
        if any(kw in text for kw in keywords):
            return mood
    return "joy"  # fallback nếu không match từ khóa nào

# Đọc file raw
with open("scripts/crawl/songs_merged.json", "r", encoding="utf-8") as f:
    songs = json.load(f)

# Thêm mood
for song in songs:
    song["mood"] = detect_mood(song.get("lyrics", ""))

# Lưu kết quả
with open("songs_with_mood.json", "w", encoding="utf-8") as f:
    json.dump(songs, f, ensure_ascii=False, indent=2)

print("✅ Đã thêm mood heuristic cho tất cả bài hát → songs_with_mood.json")
