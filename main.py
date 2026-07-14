import os
import json
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import build_default_index, BM25Index
from tools import ToolBox
from agent import CopilotAgent

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

app = FastAPI(title="Manufacturing Maintenance Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project — lock this down before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load data once at startup -------------------------------------------------------
manual_index: BM25Index = build_default_index(DATA_DIR)
with open(os.path.join(DATA_DIR, "maintenance_logs.json")) as f:
    logs: List[Dict] = json.load(f)
with open(os.path.join(DATA_DIR, "parts_catalog.json")) as f:
    parts: List[Dict] = json.load(f)

toolbox = ToolBox(manual_index, logs, parts)
agent = CopilotAgent(toolbox)  # reads ANTHROPIC_API_KEY from env

# in-memory session store: {session_id: [messages]}. Swap for Redis/DB for real deployments.
sessions: Dict[str, List[Dict]] = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_trace: List[Dict]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])
    history.append({"role": "user", "content": req.message})

    try:
        result = agent.run_turn(history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    sessions[session_id] = result["messages"]
    return ChatResponse(session_id=session_id, reply=result["final_text"], tool_trace=result["tool_trace"])


@app.post("/upload_manual")
async def upload_manual(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt/.md accepted in this demo. See README for wiring in PDF text "
                   "extraction via the pdf skill / pypdf.",
        )
    content = (await file.read()).decode("utf-8", errors="ignore")
    manual_index.add_document(content, source=file.filename)
    return {"status": "indexed", "source": file.filename, "sources_now": manual_index.list_sources()}


@app.get("/manuals")
def list_manuals():
    return {"sources": manual_index.list_sources()}


@app.get("/logs")
def list_logs():
    return {"logs": logs}


@app.get("/parts")
def list_parts():
    return {"parts": parts}


@app.get("/health")
def health():
    return {"status": "ok", "manuals_indexed": len(manual_index.list_sources())}
