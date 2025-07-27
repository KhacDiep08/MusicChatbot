import pandas as pd
import re

# Load dữ liệu
df = pd.read_csv('data/raw/lyrics_mood.csv')

# Hàm làm sạch lyrics
def clean_lyrics(text):
    text = text.lower()
    
    # Loại các section labels như [verse 1], (chorus), <bridge>
    text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', '', text)
    
    # Bỏ URL
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Chỉ giữ chữ cái, số, khoảng trắng, dấu nháy đơn
    text = re.sub(r"[^a-z0-9\s']", '', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Áp dụng hàm vào lyrics
df['clean_lyrics'] = df['lyrics'].apply(lambda x: clean_lyrics(str(x)))

# Tạo DataFrame mới
df_cleaned = df[['clean_lyrics', 'mood']].copy()

# In thử 5 dòng đầu
print(df_cleaned.head())

# Lưu file kết quả
df_cleaned.to_csv('lyrics_mood_cleaned.csv', index=False, encoding='utf-8')
print("✅ Đã lưu file lyrics_mood_cleaned.csv gồm clean_lyrics và mood.")
