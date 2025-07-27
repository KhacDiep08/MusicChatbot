import pandas as pd
import matplotlib.pyplot as plt


# Load dữ liệu
df = pd.read_csv('data/processed/mood_prediction_model/filtered_dataset.csv')

# Tổng số bài hát
total_songs = len(df)

# Tổng số nhãn mood khác nhau
unique_moods = df['mood'].nunique()

# Thống kê số lượng từng mood
mood_counts = df['mood'].value_counts()
mood_percentages = df['mood'].value_counts(normalize=True) * 100

# In kết quả
print(f"✅ Tổng số bài hát: {total_songs}")
print(f"✅ Số nhãn mood khác nhau: {unique_moods}\n")

print("📊 Phân bố số lượng theo mood:")
for mood in mood_counts.index:
    print(f"- {mood}: {mood_counts[mood]} bài ({mood_percentages[mood]:.2f}%)")

mood_counts.plot(kind='bar', color='skyblue')



plt.title("Phân bố số lượng bài hát theo mood")
plt.xlabel("Mood")
plt.ylabel("Số lượng bài hát")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
