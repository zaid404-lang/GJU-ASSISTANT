import os
import json
import hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from store import get_store

model = SentenceTransformer("all-MiniLM-L6-v2")
store = get_store(model.get_sentence_embedding_dimension())

chunks = []
ids = []
metas = []
skipped = []
subjects = set()

for root, folders, files in os.walk("pdfs"):
    folders.sort()
    for file in sorted(files):
        if not file.endswith(".pdf"):
            continue
        parts = os.path.relpath(root, "pdfs").split(os.sep)
        subject = parts[0] if parts[0] != "." else "General"
        subjects.add(subject)
        reader = PdfReader(os.path.join(root, file))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                skipped.append(subject + "/" + file + " p." + str(page_num + 1))
                continue
            for i in range(0, len(text), 600):
                chunk = text[i:i + 800]
                key = subject + "/" + file + "/" + str(page_num) + "/" + str(i)
                chunks.append(chunk)
                ids.append(hashlib.sha1(key.encode("utf-8")).hexdigest())
                metas.append({"file": file, "page": page_num + 1, "subject": subject})

if not chunks:
    print("No text found. Put text-based PDFs in the pdfs folder (one folder per subject).")
    raise SystemExit

embeddings = model.encode(chunks).tolist()
store.upsert(ids, chunks, embeddings, metas)

os.makedirs("db", exist_ok=True)
known = set()
if os.path.exists("db/subjects.json"):
    with open("db/subjects.json", encoding="utf-8") as f:
        known = set(json.load(f))
with open("db/subjects.json", "w", encoding="utf-8") as f:
    json.dump(sorted(known | subjects), f, ensure_ascii=False)

print("done", len(chunks), "chunks from", len(subjects), "subjects:", ", ".join(sorted(subjects)))

if skipped:
    print("warning:", len(skipped), "pages have no text (image only) and were skipped, for example:", skipped[:10])
