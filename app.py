
import os
import io
import time
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from logtail import LogtailHandler
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from pydantic import BaseModel

MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "20"))
LOGTAIL_SOURCE_TOKEN = os.getenv("LOGTAIL_SOURCE_TOKEN")
LOGTAIL_INGEST_HOST = os.getenv("LOGTAIL_INGEST_HOST")

logger = logging.getLogger("pdf_doc_tools_api")
logger.setLevel(logging.INFO)
logger.handlers = []

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(console)

if LOGTAIL_SOURCE_TOKEN and LOGTAIL_INGEST_HOST:
    try:
        handler = LogtailHandler(source_token=LOGTAIL_SOURCE_TOKEN, host=LOGTAIL_INGEST_HOST)
        logger.addHandler(handler)
        logger.info("Logtail handler configured.")
    except Exception as e:
        logger.error(f"Failed to configure Logtail handler: {e}")
else:
    logger.warning("LOGTAIL_SOURCE_TOKEN or LOGTAIL_INGEST_HOST not set. Skipping Logtail handler.")

app = FastAPI(title="PDF & Document Tools API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    sentences: Optional[int] = 5

def _ensure_size_ok(upload: UploadFile):
    upload.file.seek(0, io.SEEK_END)
    size = upload.file.tell()
    upload.file.seek(0)
    mb = size / (1024 * 1024)
    if mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({mb:.2f} MB). Max allowed is {MAX_FILE_MB} MB.")
    return size

def _ext_from_name_or_mime(upload: UploadFile) -> str:
    name = (upload.filename or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".txt"):
        return "txt"
    mime = (upload.content_type or "").lower()
    if "pdf" in mime:
        return "pdf"
    if "word" in mime or "docx" in mime:
        return "docx"
    if "text" in mime:
        return "txt"
    return "unknown"

def extract_text_from_upload(upload: UploadFile) -> str:
    ext = _ext_from_name_or_mime(upload)
    raw = upload.file.read()
    upload.file.seek(0)
    try:
        if ext == "pdf":
            with io.BytesIO(raw) as f:
                text = pdf_extract_text(f) or ""
        elif ext == "docx":
            with io.BytesIO(raw) as f:
                doc = Document(f)
                text = "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            text = raw.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, or TXT.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to parse document.")
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def word_count_stats(text: str):
    words = text.split()
    return {"characters": len(text), "words": len(words), "lines": len([l for l in text.splitlines() if l.strip()])}

def simple_frequency_summary(text: str, max_sentences: int = 5) -> str:
    import re, math
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    stop = set("a an the and or but is are was were be being been i you he she it we they them me my our your his her its their to of in for on at from by with as that this those these not no do does did have has had can could should would will just than then so if while during over into out up down about above below under again further here there when where why how all any both each few more most other some such nor only own same too very s t d ll m o re ve y".split())
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    freqs = {}
    for t in tokens:
        if t in stop or len(t) <= 2:
            continue
        freqs[t] = freqs.get(t, 0) + 1
    if not freqs:
        return " ".join(sentences[:max_sentences])
    scores = []
    for idx, s in enumerate(sentences):
        toks = re.findall(r"[A-Za-z']+", s.lower())
        score = sum(freqs.get(t, 0) for t in toks) / (len(toks) + 1e-9)
        score *= 1.0 + 0.1 * math.exp(-idx / 5.0)
        scores.append((score, idx, s))
    scores.sort(reverse=True)
    top = sorted(scores[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)

@app.get("/health")
def health():
    return {"ok": True, "version": "1.0.0"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    t0 = time.time()
    size = _ensure_size_ok(file)
    text = extract_text_from_upload(file)
    elapsed = time.time() - t0
    meta = {"endpoint": "/extract-text", "filename": file.filename, "ext": _ext_from_name_or_mime(file), "size_bytes": size, "elapsed_ms": int(elapsed * 1000)}
    logger.info({"event": "extract_text", **meta, "preview": text[:200]})
    return JSONResponse({"text": text, "meta": meta})

@app.post("/summary")
async def summary(file: UploadFile = File(...), sentences: int = Query(5, ge=1, le=20)):
    t0 = time.time()
    size = _ensure_size_ok(file)
    text = extract_text_from_upload(file)
    s = simple_frequency_summary(text, max_sentences=sentences)
    elapsed = time.time() - t0
    meta = {"endpoint": "/summary", "filename": file.filename, "ext": _ext_from_name_or_mime(file), "size_bytes": size, "elapsed_ms": int(elapsed * 1000), "sentences": sentences}
    logger.info({"event": "summary", **meta, "preview": s[:200]})
    return JSONResponse({"summary": s, "meta": meta})

@app.post("/word-count")
async def word_count(file: UploadFile = File(...)):
    t0 = time.time()
    size = _ensure_size_ok(file)
    text = extract_text_from_upload(file)
    stats = word_count_stats(text)
    elapsed = time.time() - t0
    meta = {"endpoint": "/word-count", "filename": file.filename, "ext": _ext_from_name_or_mime(file), "size_bytes": size, "elapsed_ms": int(elapsed * 1000)}
    logger.info({"event": "word_count", **meta, **stats})
    return JSONResponse({"counts": stats, "meta": meta})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
