import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, db_path: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        with open(db_path, 'r', encoding="utf-8") as f:
            self.db = json.load(f)

        self.embedder = SentenceTransformer(embed_model)

        self.index = self.build_faiss_index()

    def build_faiss_index(self):
      
        self.docs = [
            f"{song['title']} - {song['artist']} | {song['genre']} | {song['lyrics']} | {song['tags']}"
            for song in self.db
        ]

        embeddings = self.embedder.encode(self.docs, convert_to_numpy=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(embeddings)

        index.add(embeddings)

        return index

    def retrieve(self, query: str, top_k: int = 3):

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "doc": self.docs[idx],
                "score": float(score)
            })
        return results
