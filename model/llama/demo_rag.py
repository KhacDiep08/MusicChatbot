from rag import RAGRetriever

# Load retriever với file songs.json
rag = RAGRetriever("data/songs.json")

print("✅ RAG retriever loaded!")

# Test query
queries = [
    "bài hát nào của phoebe bridgers nói về funeral",
    "cho tôi một bài indie buồn",
    "bài hát nào thuộc thể loại rock",
]

for q in queries:
    print(f"\n🔎 Query: {q}")
    results = rag.retrieve(q, top_k=2)
    for r in results:
        print(f"  - Score: {r['score']:.4f}")
        print(f"  - Doc: {r['doc'][:200]}...")  # in 200 ký tự đầu tiên thôi cho gọn
