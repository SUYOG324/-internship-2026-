# Manufacturing Maintenance Copilot

An AI copilot for factory maintenance teams: upload equipment manuals, describe a fault, and
get a diagnosis grounded in the manual + historical logs, ranked likely causes, spare-parts
availability, and an auto-generated service report. Built to demonstrate RAG + tool-use +
industry-specific agentic workflows, not just a wrapped chatbot.

## What it actually does

- **RAG over manuals** — manuals are chunked (by section, then sliding window) and indexed
  with BM25. Every technical claim the agent makes is required (by its system prompt) to come
  from a `search_manual` or `search_maintenance_logs` tool call, not from the model's general
  knowledge — because real machines have model-specific fault codes and part numbers that a
  generic LLM will hallucinate.
- **Tool use, not just retrieval** — six tools: `search_manual`, `search_maintenance_logs`,
  `lookup_spare_parts`, `predict_failure_causes`, `create_pm_checklist`,
  `generate_service_report`. `predict_failure_causes` is the interesting one: it cross-references
  the manual's fault-code table against historical log root-causes and ranks causes by how often
  they've actually recurred on that machine type — this is the "predict common failure causes"
  requirement, implemented as frequency-weighted retrieval rather than a black-box ML model,
  which is honest about what it is and easy to defend in an interview.
- **Multi-turn agent loop** — `agent.py` runs the standard Anthropic tool-use loop (call → tool
  result → call again → ... → final text), so the copilot can chain several tool calls in one
  turn (e.g. search manual, then check parts stock, then check historical logs) before answering.
- **Service report generation** — once a diagnosis is settled in conversation, the agent calls
  `generate_service_report` to emit a structured, timestamped record — the kind of artifact a
  real maintenance team would file into a CMMS.

## Why BM25 instead of embeddings

Anthropic's API doesn't have an embeddings endpoint, so a "real" vector-embedding RAG pipeline
needs a second provider (Voyage AI, OpenAI) or a locally downloaded model. To keep this project
runnable with nothing but an `ANTHROPIC_API_KEY`, retrieval uses BM25 (Okapi) — a strong,
zero-dependency lexical ranker that's actually a good fit here, since fault codes and part
numbers are exact-match tokens anyway. `rag.py`'s `BM25Index` exposes the same `.query(text, k)`
interface a vector store would, so swapping in Chroma/Qdrant + Voyage embeddings later is a
one-file change — see the docstring at the top of `rag.py`.

## Architecture

```
manufacturing-copilot/
├── backend/
│   ├── main.py       FastAPI app: /chat, /upload_manual, /manuals, /logs, /parts
│   ├── agent.py       Claude tool-use loop (system prompt + orchestration)
│   ├── tools.py        6 tool schemas + their Python implementations
│   ├── rag.py           Chunking + BM25 retrieval engine
│   ├── requirements.txt
│   └── data/
│       ├── manuals/               2 sample manuals (hydraulic press, CNC mill)
│       ├── maintenance_logs.json  6 sample historical service records
│       └── parts_catalog.json     12 sample spare parts w/ stock & lead time
└── frontend/
    └── index.html      Single-file chat UI, tool-call trace viewer, manual upload
```

**Request flow:** user message → `/chat` → `CopilotAgent.run_turn()` → Claude decides which
tool(s) to call → `ToolBox.run()` executes against local JSON/BM25 data → results fed back to
Claude → repeat until Claude has enough to answer → final text + full tool trace returned to
the frontend (the UI shows the trace in a collapsible panel so you can see exactly what the
agent looked up — important for a maintenance tool where technicians need to trust *why*).

## Setup

```bash
cd manufacturing-copilot/backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it talks to `http://localhost:8000` by
default — editable in the sidebar).

## Try it

The sample data covers two machine types (`Hydraulic Press HP-450`, `CNC Vertical Mill VM-200`)
with fault codes, PM schedules, historical incidents, and parts. Try:

- *"HP-450-07 is throwing E-101, tonnage seems low. What should I check?"*
  → agent searches the manual for E-101, searches logs for prior E-101 incidents on that
  machine, calls `predict_failure_causes` to rank the relief-valve-drift cause it found
  historically, and checks `lookup_spare_parts` — which will flag that `HPX-PRV-14` has only
  1 in stock with a 10-day lead time, so it should tell you to order now.
- *"What's causing ALM-407 on VM200-11 and what part do I need?"*
  → same pattern on the CNC mill; the historical log points to a servo encoder, and parts
  lookup will show `VM-ENCODER-Y` is **out of stock** (18-day lead time) — a good test that the
  agent actually surfaces that constraint instead of just naming a part.
- *"Give me the monthly PM checklist for the CNC mill"*
  → `create_pm_checklist` pulls the PM schedule section from the manual.
- Upload your own `.txt`/`.md` manual via the sidebar and ask about it — it's indexed live.

## Known limitations (be upfront about these — it's a stronger portfolio piece than pretending
they don't exist)

- **PDF manuals aren't wired in yet.** `/upload_manual` only accepts `.txt`/`.md` in this demo.
  Anthropic's `pdf` skill / `pypdf` (already in requirements.txt) is the natural next step —
  extract text per page, tag each chunk with a page number, and the RAG citations become
  page-accurate.
- **Sessions are in-memory** (`sessions: Dict[str, List[Dict]]` in `main.py`) — fine for a demo,
  needs Redis/Postgres for anything persisted across restarts or multiple server instances.
- **BM25, not embeddings** — great for exact fault-code/part-number matches, weaker on
  paraphrased natural-language queries than a real semantic search. See "Why BM25" above for
  the upgrade path.
- **`predict_failure_causes` needs log volume to be useful** — with only 6 sample logs, most
  fault-code/machine-type pairs will have 0-1 historical matches. This is realistic (real
  CMMS data starts sparse too) but worth saying out loud in an interview rather than letting
  someone assume there's a trained model behind it.

## Extending it

- Swap `BM25Index` for a Chroma/Qdrant-backed embedding index (interface is identical).
- Add a `create_work_order` tool that POSTs to a real CMMS API instead of just generating JSON.
- Add streaming (Claude's SDK supports it) so the frontend shows tokens as they arrive instead
  of waiting for the full turn.
- Swap the in-memory `sessions` dict for a real store, and add auth so this can't be an open
  proxy to your Anthropic API key.
