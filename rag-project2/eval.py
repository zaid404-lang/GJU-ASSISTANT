import os
import sys
import json
import time
import random
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI, BadRequestError, RateLimitError

configs = [
    {"name": "800 chars, overlap, MiniLM (current)", "size": 800, "step": 600, "model": "all-MiniLM-L6-v2", "query_prefix": ""},
    {"name": "800 chars, no overlap, MiniLM", "size": 800, "step": 800, "model": "all-MiniLM-L6-v2", "query_prefix": ""},
    {"name": "400 chars, overlap, MiniLM", "size": 400, "step": 300, "model": "all-MiniLM-L6-v2", "query_prefix": ""},
    {"name": "1200 chars, overlap, MiniLM", "size": 1200, "step": 900, "model": "all-MiniLM-L6-v2", "query_prefix": ""},
    {"name": "800 chars, overlap, bge-small", "size": 800, "step": 600, "model": "BAAI/bge-small-en-v1.5", "query_prefix": "Represent this sentence for searching relevant passages: "},
]


def load_pages():
    pages = []
    for root, folders, files in os.walk("pdfs"):
        folders.sort()
        for file in sorted(files):
            if not file.endswith(".pdf"):
                continue
            reader = PdfReader(os.path.join(root, file))
            for num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({"file": file, "page": num + 1, "text": text})
    return pages


def make_chunks(pages, size, step):
    chunks = []
    for p in pages:
        text = p["text"]
        if not text:
            continue
        for i in range(0, len(text), step):
            chunks.append({
                "text": text[i:i + size],
                "file": p["file"],
                "page": p["page"],
                "id": p["file"] + "_" + str(p["page"]) + "_" + str(i),
            })
    return chunks


def ask_llm(llm, prompt):
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    messages = [{"role": "user", "content": prompt}]
    try:
        r = llm.chat.completions.create(model=model, messages=messages, reasoning_effort="low")
    except BadRequestError:
        r = llm.chat.completions.create(model=model, messages=messages)
    return r.choices[0].message.content.strip()


def save_questions(questions):
    os.makedirs("eval", exist_ok=True)
    with open("eval/generated.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)


def generate(count):
    pages = [p for p in load_pages() if len(p["text"]) >= 300]
    if not pages:
        print("No slides with enough text found in the pdfs folder.")
        return
    random.seed(42)
    picked = random.sample(pages, min(count, len(pages)))
    llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL", "https://api.bluesminds.com/v1"), timeout=60)

    questions = []
    if os.path.exists("eval/generated.json"):
        with open("eval/generated.json", encoding="utf-8") as f:
            questions = json.load(f)
    done = set((q["file"], q["page"]) for q in questions)
    print("already have", len(questions), "questions", flush=True)

    for p in picked:
        if (p["file"], p["page"]) in done:
            continue
        prompt = "Here is the text of one lecture slide.\n\n" + p["text"][:1500] + "\n\nWrite one short question a student could ask that this slide answers. Do not copy sentences from the slide and do not mention the slide or page. Return only the question."
        q = None
        for attempt in range(5):
            try:
                q = ask_llm(llm, prompt)
                break
            except RateLimitError:
                wait = 10 * (attempt + 1)
                print("rate limited, waiting", wait, "seconds", flush=True)
                time.sleep(wait)
            except Exception as e:
                print("skipped", p["file"], p["page"], str(e)[:80], flush=True)
                break
        if q:
            questions.append({"question": q, "file": p["file"], "page": p["page"]})
            save_questions(questions)
        print(len(questions), "/", len(picked), flush=True)
        time.sleep(3)

    print("saved", len(questions), "questions to eval/generated.json")


def load_questions():
    questions = []
    for path in ["eval/generated.json", "eval/manual.json"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                questions += json.load(f)
    return questions


def evaluate(index, config, questions, pages):
    embedder = SentenceTransformer(config["model"])
    chunks = make_chunks(pages, config["size"], config["step"])

    client = chromadb.EphemeralClient()
    collection = client.create_collection("eval_" + str(index))

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [{"file": c["file"], "page": c["page"]} for c in chunks]
    vectors = embedder.encode(texts).tolist()
    for i in range(0, len(ids), 500):
        collection.add(ids=ids[i:i + 500], documents=texts[i:i + 500], embeddings=vectors[i:i + 500], metadatas=metas[i:i + 500])

    q_texts = [config["query_prefix"] + q["question"] for q in questions]
    q_vectors = embedder.encode(q_texts).tolist()
    results = collection.query(query_embeddings=q_vectors, n_results=6)

    hits = {1: 0, 3: 0, 6: 0}
    mrr = 0
    for q, found in zip(questions, results["metadatas"]):
        rank = None
        for pos, m in enumerate(found):
            if m["file"] == q["file"] and m["page"] == q["page"]:
                rank = pos + 1
                break
        if rank:
            mrr += 1 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1

    n = len(questions)
    return {
        "name": config["name"],
        "chunks": len(chunks),
        "r1": hits[1] / n,
        "r3": hits[3] / n,
        "r6": hits[6] / n,
        "mrr": mrr / n,
    }


def run():
    questions = load_questions()
    if not questions:
        print("No questions found. Run: python eval.py generate 40")
        return
    pages = load_pages()

    rows = []
    for index, config in enumerate(configs):
        print("testing", config["name"], flush=True)
        rows.append(evaluate(index, config, questions, pages))

    lines = []
    lines.append("Questions: " + str(len(questions)) + " (top 6 chunks retrieved, a hit means the right slide is among them)")
    lines.append("")
    lines.append("| Setup | Chunks | Recall@1 | Recall@3 | Recall@6 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| " + r["name"] + " | " + str(r["chunks"]) + " | " + format(r["r1"], ".0%") + " | " + format(r["r3"], ".0%") + " | " + format(r["r6"], ".0%") + " | " + format(r["mrr"], ".2f") + " |")
    table = "\n".join(lines)

    print("\n" + table)
    os.makedirs("eval", exist_ok=True)
    with open("eval/results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nsaved to eval/results.md")


if len(sys.argv) > 1 and sys.argv[1] == "generate":
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    generate(count)
elif len(sys.argv) > 1 and sys.argv[1] == "run":
    run()
else:
    print("Usage: python eval.py generate 40   or   python eval.py run")
