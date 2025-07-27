import pandas as pd

# Bước 1: Đọc dữ liệu gốc
df = pd.read_csv("data/processed/mood_prediction_model/lyrics_mood_cleaned.csv")  

# Bước 2: Lọc 4 mood và LƯU thành file mới
df_filtered = df[df['mood'].isin(['joy', 'sadness', 'anger', 'fear'])].copy()
df_filtered.to_csv("filtered_dataset.csv", index=False)  # Lưu để dùng sau này

# Bước 3: Kiểm tra bằng cách đọc lại
df_check = pd.read_csv("filtered_dataset.csv")
print("Số samples sau lọc:", len(df_check))
print("Phân phối mới:\n", df_check['mood'].value_counts())