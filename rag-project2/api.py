import os
import re
import json
import time
import uuid
import hmac
import difflib
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI, BadRequestError
from sentence_transformers import SentenceTransformer
from store import get_store

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL", "https://api.bluesminds.com/v1")
llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
rag_key = os.getenv("RAG_API_KEY")
model_name = "GJU-Assistant"
sources_marker = "\n\n---\n**Sources:**"

if not api_key:
    raise RuntimeError("API_KEY is missing")
if not rag_key:
    raise RuntimeError("RAG_API_KEY is missing")

llm = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
store = get_store(embedder.get_sentence_embedding_dimension())
if store.count() == 0:
    raise RuntimeError("No lectures indexed. Run index.py first")

system_prompt = "You are a friendly tutor for a university course. If the student is just greeting you or chatting, reply naturally and briefly, and do not mention the slides or sources. When you use the lecture context in your answer, cite it like (file name, p. N) with the exact file name. Only cite a file if you really used it. If the slides are unclear or the student did not understand something, explain it in simple words with an example. Say clearly which part comes from the slides and which part is your own explanation. If the context has nothing related to the question, say so, then answer briefly and say that this part is not from the slides."

small_talk = ["hello", "hi", "hey", "hii", "hallo", "good morning", "good evening", "good afternoon", "thanks", "thank you", "thx", "ok", "okay", "cool", "nice", "bye", "goodbye", "how are you", "who are you", "what can you do", "help", "مرحبا", "اهلا", "أهلا", "هلا", "سلام", "السلام عليكم", "شكرا", "شكرا لك", "كيف حالك"]

app = FastAPI()


def check_auth(authorization):
    expected = "Bearer " + rag_key
    if not hmac.compare_digest(authorization or "", expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
        return text
    return ""


def ask_llm(messages):
    try:
        response = llm.chat.completions.create(model=llm_model, messages=messages, reasoning_effort="low")
    except BadRequestError:
        response = llm.chat.completions.create(model=llm_model, messages=messages)
    return response.choices[0].message.content


def open_stream(messages):
    try:
        return llm.chat.completions.create(model=llm_model, messages=messages, stream=True, reasoning_effort="low")
    except BadRequestError:
        return llm.chat.completions.create(model=llm_model, messages=messages, stream=True)


def rewrite_question(history, question):
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


subjects_cache = {"time": 0, "list": []}


def get_subjects():
    if time.time() - subjects_cache["time"] > 300:
        try:
            subjects_cache["list"] = store.load_subjects()
        except Exception:
            pass
        subjects_cache["time"] = time.time()
    return subjects_cache["list"]


def words_of(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def words_match(phrase_words, subject_words):
    for pw, sw in zip(phrase_words, subject_words):
        if pw == sw:
            continue
        if len(pw) <= 4 or len(sw) <= 4:
            return False
        if pw.startswith(sw) or sw.startswith(pw):
            continue
        if difflib.SequenceMatcher(None, pw, sw).ratio() >= 0.8:
            continue
        return False
    return True


def find_subject(text):
    words = words_of(text)
    subjects = sorted(get_subjects(), key=lambda x: -len(words_of(x)))
    for subject in subjects:
        sub_words = words_of(subject)
        n = len(sub_words)
        if n == 0:
            continue
        for i in range(len(words) - n + 1):
            if words_match(words[i:i + n], sub_words):
                return subject
    return None


def search(query, page, subject, hint):
    vector = embedder.encode([query]).tolist()[0]
    if page:
        results = store.query(vector, 6, subject or hint, page)
        if results:
            return results
    if hint and not subject:
        focused = store.query(vector, 4, hint, None)
        general = store.query(vector, 4, None, None)
        seen = set(r["text"] for r in focused)
        merged = focused + [r for r in general if r["text"] not in seen]
        return merged[:8]
    return store.query(vector, 6, subject, None)


def subject_models():
    models = {}
    if os.getenv("SUBJECT_MODELS", "false").lower() != "true":
        return models
    for s in store.load_subjects():
        models["GJU-" + s.replace(" ", "_")] = s
    return models


def clean_messages(chat_messages):
    cleaned = []
    for m in chat_messages:
        role = m.get("role")
        if role not in ["user", "assistant"]:
            continue
        text = get_text(m.get("content"))
        if role == "assistant" and sources_marker in text:
            text = text.split(sources_marker)[0]
        if text.strip():
            cleaned.append({"role": role, "content": text})
    return cleaned


def task_pieces(chat_messages):
    messages = []
    for m in chat_messages:
        messages.append({"role": m.get("role"), "content": get_text(m.get("content"))})
    yield ("answer", ask_llm(messages))


def is_small_talk(text):
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return cleaned in small_talk


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_sources(answer, results):
    found = {}
    for r in results:
        found.setdefault(r["file"], [])
        if r["page"] not in found[r["file"]]:
            found[r["file"]].append(r["page"])
    text = normalize(answer)
    parts = []
    for file, pages in found.items():
        stem = normalize(file.replace(".pdf", ""))
        if stem and stem in text:
            pages.sort()
            parts.append(file + " " + ", ".join("p." + str(n) for n in pages))
    return "; ".join(parts)


def rag_pieces(chat_messages, subject=None):
    cleaned = clean_messages(chat_messages)
    if not cleaned or cleaned[-1]["role"] != "user":
        yield ("answer", "Ask me a question about the lectures.")
        return

    question = cleaned[-1]["content"]
    history = cleaned[:-1]

    if is_small_talk(question):
        messages = [{"role": "system", "content": system_prompt}] + history[-8:] + [{"role": "user", "content": question}]
        for chunk in open_stream(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                yield ("answer", chunk.choices[0].delta.content)
        return

    start = time.time()
    yield ("thinking", "Reading your question...\n")
    page = find_page(question)
    if history:
        query = rewrite_question(history, question)
    else:
        query = question
    t_rewrite = time.time()

    hint = None
    if not subject:
        hint = find_subject(question) or find_subject(query)
    prompt = system_prompt
    scope = "all subjects"
    if subject or hint:
        scope = subject or hint
        prompt = system_prompt + " The student is asking about the subject: " + scope + "."

    yield ("thinking", "Searching " + scope + " for: " + query + "\n")
    results = search(query, page, subject, hint)
    t_search = time.time()

    context = ""
    labels = []
    for r in results:
        label = r["file"] + " p." + str(r["page"])
        context += "[" + label + "]\n" + r["text"] + "\n\n"
        if label not in labels:
            labels.append(label)

    yield ("thinking", "Found " + str(len(labels)) + " relevant slides. Writing the answer...\n")

    user_message = "Lecture context:\n" + context + "\nQuestion: " + question
    messages = [{"role": "system", "content": prompt}] + history[-8:] + [{"role": "user", "content": user_message}]

    stream = open_stream(messages)
    first = True
    full = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first:
                print("rewrite", round(t_rewrite - start, 1), "search", round(t_search - t_rewrite, 1), "first token", round(time.time() - t_search, 1), flush=True)
                first = False
            full += chunk.choices[0].delta.content
            yield ("answer", chunk.choices[0].delta.content)

    print("total", round(time.time() - start, 1), flush=True)

    sources = build_sources(full, results)
    if sources:
        yield ("answer", sources_marker + " " + sources)


def is_task(chat_messages):
    for m in reversed(chat_messages):
        if m.get("role") == "user":
            return get_text(m.get("content")).lstrip().startswith("### Task:")
    return False


def sse(pieces):
    chat_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    def make(delta, finish=None):
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return "data: " + json.dumps(data) + "\n\n"

    yield make({"role": "assistant", "content": ""})
    try:
        for kind, piece in pieces:
            if kind == "thinking":
                yield make({"reasoning_content": piece})
            else:
                yield make({"content": piece})
    except Exception:
        yield make({"content": "\n\nSomething went wrong. Please try again."})
    yield make({}, "stop")
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models(authorization: str = Header(None)):
    check_auth(authorization)
    data = [{"id": model_name, "object": "model", "created": 0, "owned_by": "gju"}]
    for model_id in sorted(subject_models()):
        data.append({"id": model_id, "object": "model", "created": 0, "owned_by": "gju"})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
def chat(body: dict, authorization: str = Header(None)):
    check_auth(authorization)
    chat_messages = body.get("messages", [])

    if is_task(chat_messages):
        pieces = task_pieces(chat_messages)
    else:
        pieces = rag_pieces(chat_messages, subject_models().get(body.get("model")))

    if body.get("stream"):
        return StreamingResponse(sse(pieces), media_type="text/event-stream")

    text = ""
    try:
        for kind, piece in pieces:
            if kind == "answer":
                text += piece
    except Exception:
        text = "Something went wrong. Please try again."

    return JSONResponse({
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })
