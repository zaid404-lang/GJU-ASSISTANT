import sys
from sentence_transformers import SentenceTransformer
from store import get_store

args = sys.argv[1:]
subject = None
if "--subject" in args:
    i = args.index("--subject")
    subject = args[i + 1]
    args = args[:i] + args[i + 2:]
question = " ".join(args) or "what is a set"

model = SentenceTransformer("all-MiniLM-L6-v2")
store = get_store(model.get_sentence_embedding_dimension())
vector = model.encode([question]).tolist()[0]

print("chunks in store:", store.count())
subjects = store.load_subjects()
print("subjects:", subjects)

print("\nDoes each subject have searchable text?")
for s in subjects:
    hits = store.query(vector, 1, s, None)
    if hits:
        print("  yes  ", s, "->", hits[0]["file"], "p." + str(hits[0]["page"]))
    else:
        print("  NO   ", s, "-> nothing found")

print("\nTop 6 for:", question, "(subject: " + str(subject or "all") + ")")
for r in store.query(vector, 6, subject, None):
    print("-", r["subject"], "|", r["file"], "p." + str(r["page"]), "|", r["text"][:110].replace("\n", " "))
