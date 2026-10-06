# GJU-Assistant: a RAG chatbot for lecture slides

Ask questions about your own lecture PDFs in the Open WebUI chat interface. Answers come from your slides, cite the file and page, and the retrieval quality is measured, not guessed.

<!-- ![Demo](docs/demo.gif) -->

## Why a custom pipeline

Open WebUI has its own document upload and knowledge features. This project uses a custom retrieval pipeline instead, so that:

- chunking, embeddings, and search are under my control and can be measured (see Evaluation)
- "explain slide 5" looks up that slide directly instead of guessing by meaning
- follow-up questions ("explain it simpler") are rewritten into standalone searches using the chat history
- sources are shown only when the answer actually cites the slides, and greetings skip retrieval entirely

## How it works

```
Browser -> Open WebUI -> api.py -> vector store (ChromaDB or Pinecone) + embedding model
                            |
                            +-> LLM provider (any OpenAI-compatible API)
```

1. `index.py` reads the PDFs, splits each page into overlapping 800-character chunks, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB.
2. `api.py` is a small FastAPI server that looks like an OpenAI-style model, so Open WebUI can talk to it. For each message it finds the closest chunks, sends them with the chat history to the LLM, and streams the answer back.
3. Open WebUI provides the interface, accounts, and chat history.

## Setup

1. Install Docker Desktop.
2. Copy `.env.example` to `.env` and fill it in:

   ```
   API_KEY=your_model_provider_key
   RAG_API_KEY=any_long_random_text
   WEBUI_SECRET_KEY=another_long_random_text
   ```

   `BASE_URL` and `LLM_MODEL` default to a Bluesminds-style OpenAI-compatible endpoint. Change them for any other provider.

3. Put your own lecture PDFs inside the `pdfs` folder, one folder per subject:

   ```
   pdfs/CS355/Lecture 1.pdf
   pdfs/Databases/Lecture 1.pdf
   ```

   PDFs placed directly in `pdfs` go into a subject called `General`.
4. Build and index:

   ```
   docker compose build
   docker compose run --rm rag-api python index.py
   ```

5. Start everything and open http://localhost:3000. The first account you create becomes the admin. Choose `GJU-Assistant`, which searches all subjects. To also get one model per subject (for example `GJU-CS355`, which searches only that subject), add `SUBJECT_MODELS=true` to `.env`.

   ```
   docker compose up -d
   ```

## Vector store

By default the slides are stored in a local ChromaDB folder (`db/`), which needs no account. For a hosted index, set these in `.env`:

```
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=gju-lectures
```

Then re-run `docker compose run --rm rag-api python index.py`. The index is created on first use (AWS us-east-1, cosine, 384 dimensions for the default embedding model). Subjects are stored as metadata and filtered at search time.

- Pinecone stores the text of each chunk, so your slides end up on a third-party server. Only use it for material you may share.
- Pinecone's free plan has limits on storage, requests, and inactivity. Check their current pricing page.
- Changing between ChromaDB and Pinecone, or adding subjects later, only needs `index.py` to be run again. Running it again does not create duplicates.
- Note: re-index after upgrading from the single-folder version, because chunks now carry a subject.

## Evaluation

Retrieval is tested on questions with a known correct slide. A hit means the right slide is among the top 6 retrieved chunks.

```
docker compose run --rm rag-api python eval.py generate 40
docker compose run --rm rag-api python eval.py run
```

`generate` asks the LLM to write one question for each of 40 random slides. You can also add your own questions in `eval/manual.json` as `[{"question": "...", "file": "exact file name.pdf", "page": 3}]`. `run` compares several chunk sizes and embedding models on the same questions and saves the table to `eval/results.md`.

Results on my course slides:

40 questions, top 6 chunks retrieved:

| Setup | Chunks | Recall@1 | Recall@3 | Recall@6 | MRR |
|---|---|---|---|---|---|
| 800 chars, overlap, MiniLM (current) | 542 | 75% | 92% | 95% | 0.84 |
| 800 chars, no overlap, MiniLM | 520 | 75% | 92% | 95% | 0.84 |
| 400 chars, overlap, MiniLM | 813 | 75% | 92% | 95% | 0.84 |
| 1200 chars, overlap, MiniLM | 518 | 72% | 92% | 95% | 0.82 |
| 800 chars, overlap, bge-small | 542 | 75% | 92% | 95% | 0.84 |

Chunk size, overlap, and the embedding model made almost no difference: every setup found the right slide in the top 6 for 38 of 40 questions, and the only gap (72% vs 75% Recall@1) is a single question. With 40 questions that is noise, so I kept the current setup (MiniLM, 800 characters with overlap).

## Limitations

- Questions written by an LLM from a slide tend to share words with it, so the scores are higher than real student questions would get. Add manual questions for a harder test.
- Slides that are only images have no text. `index.py` prints a warning with the skipped pages, but they cannot be searched.
- Answer speed depends on the model provider. Reasoning models can take several seconds before the first word.
- The default embedding model is English-focused.
- Sources are shown only if the model cites a file in its answer. If it uses the slides without citing, none appear.

## Notes

- Never upload `.env`. It is already in `.gitignore`, together with `db/`, `pdfs/`, and the generated question files. It also holds your Pinecone key if you use it.
- Do not publish course material you do not have the right to share. Each person adds their own PDFs.
- New sign-ups in Open WebUI need the admin's approval (Admin Panel > Users).
- Open WebUI saves its connection settings on the first start. If you change `RAG_API_KEY` later, update it in Admin Panel > Settings > Connections.
- `app.py` and `ask.py` are older Streamlit and terminal versions and are optional.
