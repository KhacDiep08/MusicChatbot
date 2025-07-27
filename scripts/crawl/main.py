from crawl_songs import crawl_from_file

def main():
    input_file = "az_urls.txt"       # Danh sách URL AZLyrics mỗi bài 1 dòng
    output_file = "songs.json"    # Tệp JSON chứa kết quả sau khi crawl
    crawl_from_file(input_file, output_file)

if __name__ == "__main__":
    main()
