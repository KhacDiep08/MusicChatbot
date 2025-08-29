import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, db_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load music database (JSON file)
        with open(db_path, 'r', encoding="utf-8") as f:
            self.db = json.load(f)

        # Embedding model (Sentence-BERT)
        self.embedder = SentenceTransformer(embed_model)

        # Build FAISS index (vector DB)
        self.index = self.build_faiss_index()

    def build_faiss_index(self):
        """
        Tạo FAISS index từ lyrics + metadata
        """
        # Ghép tất cả thông tin thành 1 document string
        self.docs = [
            f"{song['title']} - {song['artist']} | {song['genre']} | {song['lyrics']}"
            for song in self.db
        ]

        # Encode thành vector
        embeddings = self.embedder.encode(self.docs, convert_to_numpy=True)

        # FAISS Index với cosine similarity (dùng inner product sau khi normalize)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        # Normalize embeddings để dùng cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vào index
        index.add(embeddings)

        return index

    def retrieve(self, query: str, top_k: int = 3):
        """
        Truy vấn dữ liệu từ câu hỏi
        """
        # Encode query
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # Tìm top-k docs
        scores, indices = self.index.search(q_emb, top_k)

        # Gom kết quả
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "doc": self.docs[idx],
                "score": float(score)
            })
        return results
