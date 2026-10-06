import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

st.set_page_config(page_title="Lecture Tutor", page_icon="📚")

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL", "https://api.bluesminds.com/v1")
llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
access_code = os.getenv("ACCESS_CODE")

system_prompt = "You are a friendly tutor for a university course. Use the lecture context as your main source and mention the file and page. If the slides are unclear or the student did not understand something, explain it in simple words with an example. Say clearly which part comes from the slides and which part is your own explanation. If the context has nothing related to the question, say so, then answer briefly and say that this part is not from the slides."

if not api_key:
    st.error("API_KEY is missing. Add it to your .env file or your host's secrets.")
    st.stop()

if access_code:
    if "unlocked" not in st.session_state:
        st.session_state.unlocked = False
    if not st.session_state.unlocked:
        st.title("📚 Lecture Tutor")
        code = st.text_input("Access code", type="password")
        if code:
            if code == access_code:
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Wrong code")
        st.stop()


@st.cache_resource
def load_tools():
    api = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection("lectures")
    return api, embedder, collection


try:
    api, embedder, collection = load_tools()
except Exception:
    st.error("The lecture database was not found. Run python index.py first.")
    st.stop()


def ask_llm(messages):
    response = api.chat.completions.create(model=llm_model, messages=messages)
    return response.choices[0].message.content


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


def search(query, page):
    q_vec = embedder.encode([query]).tolist()
    if page:
        results = collection.query(query_embeddings=q_vec, n_results=6, where={"page": page})
        if results["documents"][0]:
            return results
    return collection.query(query_embeddings=q_vec, n_results=6)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 Lecture Tutor")
    st.write("Ask about anything in the lecture slides.")
    st.write("Tips:")
    st.write("- Say the slide number: *explain slide 5*")
    st.write("- Ask follow-ups: *explain it simpler*")
    st.write("- Ask for examples: *give me an example*")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Lecture Tutor")
st.caption("Answers come from the lecture slides, with the file and page.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(s)

question = st.chat_input("Ask about the lectures...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                page = find_page(question)
                query = rewrite_question(history, question)
                results = search(query, page)

                context = ""
                sources = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    label = meta["file"] + " p." + str(meta["page"])
                    context += "[" + label + "]\n" + doc + "\n\n"
                    if label not in sources:
                        sources.append(label)

                user_message = "Lecture context:\n" + context + "\nQuestion: " + question
                messages = [{"role": "system", "content": system_prompt}] + history[-8:] + [{"role": "user", "content": user_message}]
                answer = ask_llm(messages)
            except Exception:
                st.error("Something went wrong. Please try again.")
                st.stop()

        st.markdown(answer)
        with st.expander("Sources"):
            for s in sources:
                st.write(s)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
