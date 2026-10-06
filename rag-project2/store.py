import os
import json
import hashlib

store_type = os.getenv("VECTOR_STORE", "chroma").strip().lower()


class ChromaStore:
    def __init__(self):
        import chromadb
        client = chromadb.PersistentClient(path="db")
        self.collection = client.get_or_create_collection("lectures")

    def count(self):
        return self.collection.count()

    def load_subjects(self):
        if os.path.exists("db/subjects.json"):
            with open("db/subjects.json", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_subjects(self, subjects):
        os.makedirs("db", exist_ok=True)
        known = set(self.load_subjects())
        with open("db/subjects.json", "w", encoding="utf-8") as f:
            json.dump(sorted(known | set(subjects)), f, ensure_ascii=False)

    def upsert(self, ids, texts, vectors, metas):
        for i in range(0, len(ids), 500):
            self.collection.upsert(
                ids=ids[i:i + 500],
                documents=texts[i:i + 500],
                embeddings=vectors[i:i + 500],
                metadatas=metas[i:i + 500],
            )

    def query(self, vector, k, subject, page):
        conditions = []
        if subject:
            conditions.append({"subject": subject})
        if page:
            conditions.append({"page": page})
        kwargs = {}
        if len(conditions) == 1:
            kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            kwargs["where"] = {"$and": conditions}
        results = self.collection.query(query_embeddings=[vector], n_results=k, **kwargs)
        found = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            found.append({"text": text, "file": meta["file"], "page": meta["page"], "subject": meta.get("subject", "")})
        return found


class PineconeStore:
    def __init__(self, dimension):
        from pinecone import Pinecone, ServerlessSpec
        key = os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY is missing")
        name = os.getenv("PINECONE_INDEX", "gju-lectures")
        pc = Pinecone(api_key=key)
        if not pc.has_index(name):
            pc.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = pc.Index(name)
        self.namespace = "lectures"
        self.dimension = dimension

    def load_subjects(self):
        vector = [1.0] + [0.0] * (self.dimension - 1)
        response = self.index.query(vector=vector, top_k=100, include_metadata=True, namespace="subjects")
        return sorted(match.metadata["subject"] for match in response.matches)

    def save_subjects(self, subjects):
        records = []
        for s in subjects:
            vector = [1.0] + [0.0] * (self.dimension - 1)
            records.append({"id": hashlib.sha1(s.encode("utf-8")).hexdigest(), "values": vector, "metadata": {"subject": s}})
        for i in range(0, len(records), 100):
            self.index.upsert(vectors=records[i:i + 100], namespace="subjects")

    def count(self):
        stats = self.index.describe_index_stats()
        return stats.total_vector_count

    def upsert(self, ids, texts, vectors, metas):
        records = []
        for id_, text, vector, meta in zip(ids, texts, vectors, metas):
            metadata = dict(meta)
            metadata["text"] = text
            records.append({"id": id_, "values": vector, "metadata": metadata})
        for i in range(0, len(records), 100):
            self.index.upsert(vectors=records[i:i + 100], namespace=self.namespace)

    def query(self, vector, k, subject, page):
        metadata_filter = {}
        if subject:
            metadata_filter["subject"] = {"$eq": subject}
        if page:
            metadata_filter["page"] = {"$eq": page}
        kwargs = {}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        response = self.index.query(vector=vector, top_k=k, include_metadata=True, namespace=self.namespace, **kwargs)
        found = []
        for match in response.matches:
            meta = match.metadata
            found.append({"text": meta["text"], "file": meta["file"], "page": int(meta["page"]), "subject": meta.get("subject", "")})
        return found


def get_store(dimension):
    if store_type == "pinecone":
        store = PineconeStore(dimension)
        print("Vector store: Pinecone index", os.getenv("PINECONE_INDEX", "gju-lectures"), flush=True)
        return store
    if store_type == "chroma":
        print("Vector store: local ChromaDB (data stays on this computer)", flush=True)
        return ChromaStore()
    raise RuntimeError("VECTOR_STORE must be chroma or pinecone, not " + store_type)
