import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

def normalize(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower()

class Retriever:
    def __init__(self, documents):
        self.docs = [normalize(d) for d in documents]

        self.tokenized = [d.split() for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)

        self.embeddings = model.encode(self.docs, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, top_k=1):
        query = normalize(query)
        tokens = query.split()

        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:50]

        qvec = model.encode([query], normalize_embeddings=True)
        _, faiss_idx = self.index.search(qvec, 50)

        candidates = set(bm25_top) | set(faiss_idx[0])

        scored = []
        for i in candidates:
            score = bm25_scores[i]
            scored.append((score, self.docs[i]))

        scored.sort(reverse=True)
        return scored[:top_k]
