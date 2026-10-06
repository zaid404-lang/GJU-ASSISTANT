import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL", "https://api.bluesminds.com/v1")
llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

if not api_key:
    print("API_KEY is missing. Create a .env file (see .env.example).")
    raise SystemExit

api = OpenAI(api_key=api_key, base_url=base_url, timeout=60)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db")
collection = client.get_collection("lectures")

system_prompt = "You are a friendly tutor for a university course. Use the lecture context as your main source and mention the file and page. If the slides are unclear or the student did not understand something, explain it in simple words with an example. Say clearly which part comes from the slides and which part is your own explanation. If the context has nothing related to the question, say so, then answer briefly and say that this part is not from the slides."

history = []


def ask_llm(messages):
    response = api.chat.completions.create(model=llm_model, messages=messages)
    return response.choices[0].message.content


def rewrite_question(question):
    text = ""
    for m in history[-6:]:
        text += m["role"] + ": " + m["content"] + "\n"
    prompt = "Rewrite the last student question as one clear standalone search query for lecture slides. Use the conversation to fill in missing words like 'it' or 'that'. If the question is vague, add the likely topic keywords. Return only the query.\n\n" + text + "Last question: " + question
    try:
        new_query = ask_llm([{"role": "user", "content": prompt}]).strip()
        if new_query:
            return new_query
    except Exception:
        pass
    return question


def find_page(question):
    match = re.search(r"(?:slide|page)\s*(\d+)", question.lower())
    if match:
        return int(match.group(1))
    return None


def search(query, page):
    q_vec = model.encode([query]).tolist()
    if page:
        results = collection.query(query_embeddings=q_vec, n_results=6, where={"page": page})
        if results["documents"][0]:
            return results
    return collection.query(query_embeddings=q_vec, n_results=6)


while True:
    question = input("\nQuestion (type exit to stop): ").strip()
    if question == "":
        continue
    if question.lower() == "exit":
        break

    page = find_page(question)
    query = rewrite_question(question)
    results = search(query, page)

    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += "[" + meta["file"] + " p." + str(meta["page"]) + "]\n" + doc + "\n\n"

    user_message = "Lecture context:\n" + context + "\nQuestion: " + question

    messages = [{"role": "system", "content": system_prompt}] + history[-8:] + [{"role": "user", "content": user_message}]

    try:
        answer = ask_llm(messages)
    except Exception as e:
        print("Error:", e)
        continue

    print("\n" + answer)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
