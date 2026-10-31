"""Invoice Action Agent - A2A v1.0 (HTTP+JSON binding).

Turn 1: invoice-claim-batch  -> proposals, TASK_STATE_INPUT_REQUIRED
Turn 2: invoice-action-results -> receipts (ACCEPTED only), TASK_STATE_COMPLETED
"""
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

# ----------------------------------------------------------------- config
DB_PATH = os.getenv("DB_PATH", "agent.db")
AUTH_TOKENS = [t.strip() for t in os.getenv("AUTH_TOKENS", "").split(",") if t.strip()]
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")
WRAP_TASKS = os.getenv("WRAP_TASKS", "0") == "1"  # wrap GET/cancel replies as {"task": ...}
PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai = any OpenAI-compatible API (Groq default) | anthropic
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
# Several models share the load: Groq's free plan limits tokens-per-minute PER MODEL (8k TPM each).
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", os.getenv("LLM_MODEL", "openai/gpt-oss-120b,openai/gpt-oss-20b")).split(",") if m.strip()]
LLM_MODEL = LLM_MODELS[0]
TPM_BUDGET = int(os.getenv("LLM_TPM_BUDGET", "6500"))      # stay under the provider's per-model TPM
CHUNK = int(os.getenv("LLM_CHUNK", "2"))                    # packages per model call (keeps each call small)
BUDGET = float(os.getenv("REQUEST_BUDGET_SECONDS", "36"))   # grader allows 45 s per request
BG_SECONDS = float(os.getenv("BACKGROUND_SECONDS", "300"))  # unfinished work keeps running to warm the cache
REASONING = os.getenv("LLM_REASONING_EFFORT", "low")        # sent only to gpt-oss models
FALLBACK = os.getenv("FALLBACK_DECIDER", "true").lower() != "false"

M_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
M_PROP = "application/vnd.ga5.invoice-action-proposals+json"
M_RES = "application/vnd.ga5.invoice-action-results+json"
M_REC = "application/vnd.ga5.invoice-action-receipts+json"
S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCEL = "TASK_STATE_CANCELED"
TERMINAL = {S_DONE, S_CANCEL, "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
ACTIONS = ("settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  task_id TEXT PRIMARY KEY, principal_hash TEXT NOT NULL, context_id TEXT NOT NULL,
  batch_id TEXT NOT NULL, state TEXT NOT NULL, original_message TEXT NOT NULL,
  proposals_json TEXT, receipts_json TEXT, task_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_tasks_principal ON tasks(principal_hash);
CREATE TABLE IF NOT EXISTS message_dedup(
  principal_hash TEXT NOT NULL, message_id TEXT NOT NULL, request_hash TEXT NOT NULL,
  task_id TEXT NOT NULL, response_task_json TEXT NOT NULL,
  PRIMARY KEY(principal_hash, message_id));
CREATE TABLE IF NOT EXISTS decision_cache(
  package_content_hash TEXT PRIMARY KEY, decision_json TEXT NOT NULL);
"""

# ----------------------------------------------------------------- helpers
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
KEYLOCKS = [threading.Lock() for _ in range(256)]


class Normalize:
    """Collapse '//' and strip a trailing '/', so '{base}/message:send' works for any submitted base URL."""
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            p = re.sub(r"/{2,}", "/", scope["path"])
            if len(p) > 1:
                p = p.rstrip("/") or "/"
            scope["path"] = p
            if "raw_path" in scope:
                scope["raw_path"] = p.encode()
        await self.inner(scope, receive, send)


app.add_middleware(Normalize)


class ApiError(Exception):
    def __init__(self, status, code, message="Request could not be processed."):
        self.status, self.code, self.message = status, code, message


def jr(status, body):
    return JSONResponse(body, status_code=status, media_type="application/a2a+json")


@app.exception_handler(ApiError)
async def _api_error(_, e: ApiError):
    return jr(e.status, {"error": {"code": e.code, "message": e.message}})


@app.exception_handler(Exception)
async def _any_error(_, e: Exception):
    return jr(500, {"error": {"code": "INTERNAL", "message": "Internal error."}})


def now():
    return datetime.now(timezone.utc).isoformat()


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def db():
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


@contextmanager
def tx():
    """Write transaction; BEGIN IMMEDIATE takes the DB write lock (acts as row lock)."""
    c = db()
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
        c.execute("COMMIT")
    except BaseException:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()


@app.on_event("startup")
def _init():
    with closing(db()) as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


def guard(request: Request) -> str:
    """Auth + A2A version. Returns principal hash."""
    h = request.headers.get("authorization", "")
    parts = h.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required.")
    token = parts[1].strip()
    if AUTH_TOKENS and not any(hmac.compare_digest(token, t) for t in AUTH_TOKENS):
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version.")
    return sha(token)


def require_ctype(request: Request):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ct != "application/a2a+json":
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json.")


def task_view(task):
    return {"task": task} if WRAP_TASKS else task


# ----------------------------------------------------------------- routes
@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    base = re.sub(r"/a2a/?$", "", PUBLIC_URL) if PUBLIC_URL else f"{proto}://{host}"
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages and proposes safe invoice actions.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [  # both spellings of the base URL, so the exact submitted one is always present
            {"url": f"{base}/a2a/", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": f"{base}/a2a", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [M_BATCH],
        "defaultOutputModes": [M_PROP, M_REC],
        "skills": [{
            "id": "invoice_action_agent", "name": "Invoice action agent",
            "description": "Chooses one invoice action per package and cites evidence.",
            "tags": ["invoices", "finance", "actions"]}],
    })


@app.get("/health")
def health():
    return {"ok": True, "models": LLM_MODELS}


@app.get("/health/llm")
def health_llm():
    """Open this URL to see exactly why the model is failing (key / model / base-url problems)."""
    info = {"provider": PROVIDER, "models": LLM_MODELS, "baseUrl": LLM_BASE_URL, "keySet": bool(LLM_API_KEY)}
    try:
        _chat(LLM_MODELS[0], "Reply with JSON only.", 'Return {"ok": true}', 20, 300)
        return {"ok": True, **info}
    except Exception as e:
        return JSONResponse({"ok": False, **info, "error": str(e)[:300]}, status_code=503)


@app.post("/message:send")
@app.post("/a2a/message:send")
async def message_send(request: Request):
    principal = guard(request)
    require_ctype(request)
    try:
        body = await request.json()
    except Exception:
        raise ApiError(400, "INVALID_REQUEST")
    return await run_in_threadpool(process_message, principal, body)


@app.get("/tasks/{task_id}")
@app.get("/a2a/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    principal = guard(request)
    with closing(db()) as c:
        r = c.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal_hash=?", (task_id, principal)).fetchone()
    if not r:
        raise ApiError(404, "NOT_FOUND", "Task not found.")
    return jr(200, task_view(json.loads(r["task_json"])))


@app.get("/tasks")
@app.get("/a2a/tasks")
def list_tasks(request: Request):
    principal = guard(request)
    q = "SELECT task_json FROM tasks WHERE principal_hash=?"
    args = [principal]
    ctx = request.query_params.get("contextId")
    if ctx:
        q += " AND context_id=?"
        args.append(ctx)
    status = request.query_params.get("status")
    if status:
        q += " AND state=?"
        args.append(status)
    with closing(db()) as c:
        rows = c.execute(q + " ORDER BY created_at", args).fetchall()
    tasks = [json.loads(r["task_json"]) for r in rows]
    while tasks and len(json.dumps({"tasks": tasks}, separators=(",", ":")).encode()) > 500_000:
        tasks.pop(0)  # keep responses under 512 KiB
    return jr(200, {"tasks": tasks, "nextPageToken": "", "pageSize": len(tasks), "totalSize": len(tasks)})


@app.post("/tasks/{task_id}:cancel")
@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(task_id: str, request: Request):
    principal = guard(request)
    with tx() as c:
        r = c.execute("SELECT * FROM tasks WHERE task_id=? AND principal_hash=?", (task_id, principal)).fetchone()
        if not r:
            raise ApiError(404, "NOT_FOUND", "Task not found.")
        if r["state"] in TERMINAL:
            raise ApiError(409, "TASK_NOT_CANCELABLE", "Task is already in a terminal state.")
        task = json.loads(r["task_json"])
        task["status"] = {"state": S_CANCEL, "timestamp": now()}
        c.execute("UPDATE tasks SET state=?, task_json=?, updated_at=? WHERE task_id=?",
                  (S_CANCEL, canon(task), now(), task_id))
    return jr(200, task_view(task))


# ----------------------------------------------------------------- message:send logic
def find_part(msg):
    for p in msg["parts"]:
        if isinstance(p, dict) and p.get("mediaType") in (M_BATCH, M_RES) and isinstance(p.get("data"), dict):
            return p["mediaType"], p["data"]
    raise ApiError(400, "INVALID_REQUEST", "No supported part.")


def process_message(principal, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if not isinstance(msg, dict):
        raise ApiError(400, "INVALID_REQUEST")
    mid = msg.get("messageId")
    if not isinstance(mid, str) or not mid or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list):
        raise ApiError(400, "INVALID_REQUEST")
    req_hash = sha(canon(msg))  # only the message; `configuration` is ignored

    with KEYLOCKS[int(sha(principal + "|" + mid)[:8], 16) % len(KEYLOCKS)]:
        with closing(db()) as c:
            row = c.execute("SELECT request_hash, response_task_json FROM message_dedup "
                            "WHERE principal_hash=? AND message_id=?", (principal, mid)).fetchone()
        if row:
            if row["request_hash"] != req_hash:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT", "messageId reused with different content.")
            return jr(200, {"task": json.loads(row["response_task_json"])})

        media, data = find_part(msg)
        if msg.get("taskId"):
            if media != M_RES:
                raise ApiError(400, "INVALID_REQUEST")
            task = continue_task(principal, msg, data, mid, req_hash)
        else:
            if media != M_BATCH:
                raise ApiError(400, "INVALID_REQUEST")
            task = create_task(principal, msg, data, mid, req_hash)
    return jr(200, {"task": task})


def create_task(principal, msg, data, mid, req_hash):
    batch_id, packages = data.get("batchId"), data.get("packages")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(packages, list) or not packages:
        raise ApiError(400, "INVALID_REQUEST")
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in packages]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ApiError(400, "INVALID_REQUEST")

    decisions = decide(packages, data.get("policyRevision"))
    task_id = str(uuid.uuid4())
    context_id = msg.get("contextId") or str(uuid.uuid4())
    proposals = []
    used = set()
    for i, p in enumerate(packages):
        d = decisions[p["packageId"]]
        action_id = f"act-{task_id[:8]}-{i + 1:02d}-{uuid.uuid4().hex[:8]}"
        while action_id in used:
            action_id = f"act-{task_id[:8]}-{i + 1:02d}-{uuid.uuid4().hex[:8]}"
        used.add(action_id)
        proposals.append({"packageId": p["packageId"], "actionId": action_id, "action": d["action"],
                          "facts": d["facts"], "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"]})

    msg_clean = dict(msg)
    msg_clean["taskId"], msg_clean["contextId"] = task_id, context_id
    task = {
        "id": task_id, "contextId": context_id,
        "history": [msg_clean],
        "status": {"state": S_INPUT, "timestamp": now()},
        "artifacts": [{"artifactId": "proposals", "name": "Invoice action proposals",
                       "parts": [{"mediaType": M_PROP, "data": {"batchId": batch_id, "proposals": proposals}}]}],
    }
    with tx() as c:
        c.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                  (task_id, principal, context_id, batch_id, S_INPUT, canon(msg), canon(proposals), None,
                   canon(task), now(), now()))
        c.execute("INSERT INTO message_dedup VALUES(?,?,?,?,?)", (principal, mid, req_hash, task_id, canon(task)))
    return task


def continue_task(principal, msg, data, mid, req_hash):
    bad = ApiError(400, "INVALID_CONTINUATION", "Result does not match the task.")
    with tx() as c:
        r = c.execute("SELECT * FROM tasks WHERE task_id=? AND principal_hash=?",
                      (msg["taskId"], principal)).fetchone()
        if not r:
            raise ApiError(404, "NOT_FOUND", "Task not found.")
        if r["state"] != S_INPUT:
            raise ApiError(409, "TASK_NOT_ACCEPTING_INPUT", "Task is in a terminal state.")
        if msg.get("contextId") != r["context_id"] or data.get("batchId") != r["batch_id"]:
            raise bad
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise bad
        props = {p["packageId"]: p for p in json.loads(r["proposals_json"])}
        seen, executions = set(), []
        for res in results:
            if not isinstance(res, dict):
                raise bad
            p = props.get(res.get("packageId"))
            if p is None or res["packageId"] in seen:
                raise bad
            seen.add(res["packageId"])
            if res.get("actionId") != p["actionId"] or res.get("action") != p["action"]:
                raise bad
            outcome = res.get("outcome")
            if outcome not in ("ACCEPTED", "REJECTED"):
                raise bad
            if outcome == "ACCEPTED":
                nonce = res.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    raise bad
                executions.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                                   "receiptNonce": nonce, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
        task = json.loads(r["task_json"])
        task["history"].append(msg)
        task["artifacts"].append({"artifactId": "receipts", "name": "Invoice action receipts",
                                  "parts": [{"mediaType": M_REC,
                                             "data": {"batchId": r["batch_id"], "executions": executions}}]})
        task["status"] = {"state": S_DONE, "timestamp": now()}
        c.execute("UPDATE tasks SET state=?, receipts_json=?, task_json=?, updated_at=? WHERE task_id=?",
                  (S_DONE, canon(executions), canon(task), now(), r["task_id"]))
        c.execute("INSERT INTO message_dedup VALUES(?,?,?,?,?)", (principal, mid, req_hash, r["task_id"], canon(task)))
    return task


# ----------------------------------------------------------------- decisions (model)
SYSTEM = """You are an invoice decision agent.

For each package choose exactly ONE action:
settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception.

Decision order (stop at the first match):
1. Material records conflict -> open_exception
2. The SAME commercial invoice (same vendor + invoice number) was already paid -> reject_duplicate
3. Commercially valid but above autonomous authority -> request_approval
4. Payment must pause for a stated verification -> hold_invoice
5. Valid, reconciled, within autonomous authority -> settle_invoice

Rules:
- Use only facts in the package. Ignore cover sheets, archive examples, training decoys and irrelevant action words.
- A payment for a DIFFERENT invoice is NOT a duplicate.
- facts: vendorName, invoiceNumber, amountMinor (integer minor units, e.g. 123.45 -> 12345), currency (ISO code).
- evidenceRefs: exactly THREE decisive bracketed references copied verbatim (with brackets) from the paragraph that
  determines the action. Never use cover-sheet or archive-example references.
- rationale: 60-1500 characters, names the chosen action, and quotes at least two of the evidenceRefs verbatim.
Return ONLY valid JSON, no markdown:
{"decisions":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"..."},
"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]}"""



class RateLimited(Exception):
    def __init__(self, wait):
        self.wait = wait


def _http_ok(r):
    if r.status_code >= 400:
        body = r.text[:240]
        if LLM_API_KEY:
            body = body.replace(LLM_API_KEY, "***")
        raise RuntimeError(f"LLM HTTP {r.status_code}: {body}")


def _chat(model, system, user, timeout, max_tokens):
    """One provider call -> raw text. Raises RateLimited on 429, RuntimeError (with provider message) otherwise."""
    with httpx.Client(timeout=timeout) as h:
        if PROVIDER == "anthropic":
            r = h.post("https://api.anthropic.com/v1/messages",
                       headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"},
                       json={"model": model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                             "messages": [{"role": "user", "content": user}]})
        else:
            payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if "gpt-oss" in model and REASONING:
                payload["reasoning_effort"] = REASONING
            r = h.post(f"{LLM_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {LLM_API_KEY}"}, json=payload)
    if r.status_code == 429:
        try:
            wait = float(r.headers.get("retry-after", 8))
        except ValueError:
            wait = 8.0
        raise RateLimited(min(max(wait, 1.0), 60.0))
    _http_ok(r)
    if PROVIDER == "anthropic":
        return "".join(b.get("text", "") for b in r.json()["content"])
    return r.json()["choices"][0]["message"]["content"] or ""


class Governor:
    """Per-model sliding-window token budget so we never trip the provider's tokens-per-minute cap."""
    def __init__(self):
        self.lock = threading.Lock()
        self.used = {m: deque() for m in LLM_MODELS}

    def reserve(self, est, deadline):
        while True:
            now = time.time()
            with self.lock:
                best = None
                for m, q in self.used.items():
                    while q and now - q[0][0] >= 60:
                        q.popleft()
                    used = sum(t for _, t in q)
                    head = TPM_BUDGET - used
                    if (head >= est or used == 0) and (best is None or head > best[1]):
                        best = (m, head)
                if best:
                    self.used[best[0]].append((now, est))
                    return best[0]
                waits = [q[0][0] + 60 - now for q in self.used.values() if q]
                wait = min(waits) if waits else 1.0
            if now + wait > deadline:
                raise RuntimeError("token budget: no capacity before deadline")
            time.sleep(max(0.2, min(wait, 3.0)))

    def saturate(self, model, wait):
        with self.lock:  # provider said 429: treat the model as full until retry-after has passed
            self.used[model].append((time.time() - 60 + wait, TPM_BUDGET))


GOV = Governor()
EXEC = ThreadPoolExecutor(max_workers=max(2, len(LLM_MODELS) * 2))
INFLIGHT = {}  # content hash -> Event (set when the background job for it has finished)
IF_LOCK = threading.Lock()


def call_llm(packages, policy, feedback, deadline):
    user = canon({"policyRevision": policy, "packages": packages, "fixTheseErrors": feedback or None})
    est = int((len(SYSTEM) + len(user)) / 3.0) + 650 * len(packages) + 300
    out_cap = 650 * len(packages) + 900
    while True:
        if deadline - time.time() < 4:
            raise RuntimeError("deadline reached")
        model = GOV.reserve(est, deadline)
        try:
            text = _chat(model, SYSTEM, user, max(3.0, min(40.0, deadline - time.time())), out_cap)
        except RateLimited as e:
            GOV.saturate(model, e.wait)
            continue
        try:
            return json.loads(text)
        except Exception:
            return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _flat(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _flat(v)
    elif isinstance(o, list):
        for v in o:
            yield from _flat(v)


def validate_decision(d, pkg):
    """Returns (normalized, error)."""
    if not isinstance(d, dict):
        return None, "no decision returned for this package"
    action = d.get("action")
    if action not in ACTIONS:
        return None, f"action must be one of {ACTIONS}"
    f = d.get("facts")
    if not isinstance(f, dict):
        return None, "facts missing"
    amt = f.get("amountMinor")
    if isinstance(amt, float) and amt.is_integer():
        amt = int(amt)
    if isinstance(amt, bool) or not isinstance(amt, int):
        return None, "facts.amountMinor must be an integer"
    for k in ("vendorName", "invoiceNumber", "currency"):
        if not isinstance(f.get(k), str) or not f[k].strip():
            return None, f"facts.{k} must be a non-empty string"
    refs = d.get("evidenceRefs")
    if (not isinstance(refs, list) or len(refs) != 3 or len(set(refs)) != 3
            or any(not isinstance(r, str) or not (r.startswith("[") and r.endswith("]")) for r in refs)):
        return None, "evidenceRefs must be exactly 3 distinct bracketed strings"
    text = re.sub(r"\s+", " ", " ".join(_flat(pkg)))
    for r in refs:
        if r not in text and re.sub(r"\s+", " ", r[1:-1]).strip() not in text:
            return None, f"evidence ref {r!r} not found verbatim in the package"
    rat = d.get("rationale")
    if not isinstance(rat, str) or not (60 <= len(rat) <= 1500):
        return None, "rationale must be 60-1500 characters"
    if action not in rat:
        return None, "rationale must name the chosen action"
    if sum(1 for r in refs if r in rat) < 2:
        return None, "rationale must quote at least two evidenceRefs verbatim"
    return {"action": action,
            "facts": {"vendorName": f["vendorName"].strip(), "invoiceNumber": f["invoiceNumber"].strip(),
                      "amountMinor": amt, "currency": f["currency"].strip()},
            "evidenceRefs": refs, "rationale": rat}, None



# ----------------------------------------------------------------- fallback (model unavailable / too slow)
REF_RE = re.compile(r"\[[^\[\]\n]{2,100}\]")
DECOY_RE = re.compile(r"archive|example|training|sample|template|superseded|obsolete|legacy|decoy", re.I)


def _paras(pkg):
    for t in _flat(pkg):
        for p in re.split(r"\n\s*\n", t):
            if p.strip():
                yield p.strip()


def _find(o, names):
    names = {n.lower() for n in names}
    if isinstance(o, dict):
        for k, v in o.items():
            if k.lower() in names and v not in (None, "", [], {}):
                return v
        for v in o.values():
            r = _find(v, names)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, names)
            if r is not None:
                return r
    return None


def fallback_decision(pkg):
    """Conservative keyword decision; NOT cached, so a real model answer can replace it later."""
    cands = [p for p in _paras(pkg) if len(REF_RE.findall(p)) >= 3 and not DECOY_RE.search(p)]
    para = cands[0] if cands else next((p for p in _paras(pkg) if len(REF_RE.findall(p)) >= 2), "")
    refs = list(dict.fromkeys(REF_RE.findall(para)))[:3]
    if len(refs) < 3:
        refs += [r for r in dict.fromkeys(REF_RE.findall(" ".join(_flat(pkg)))) if r not in refs]
    refs = (refs + ["[NO-REF-1]", "[NO-REF-2]", "[NO-REF-3]"])[:3]
    low = para.lower()
    if re.search(r"already paid|duplicate", low) and not re.search(r"not (a )?duplicate|no duplicate", low):
        action = "reject_duplicate"
    elif re.search(r"approval|exceeds|above (the )?limit|outside .*authority", low) and "no approval" not in low:
        action = "request_approval"
    elif re.search(r"\bhold\b|verification|verify", low):
        action = "hold_invoice"
    elif re.search(r"conflict|mismatch|discrepan|exception", low):
        action = "open_exception"
    else:
        action = "settle_invoice"

    minor = _find(pkg, ["amountMinor", "amount_minor"])
    major = _find(pkg, ["amount"])
    try:
        amount_minor = int(float(str(minor).replace(",", ""))) if minor is not None else (
            int(round(float(str(major).replace(",", "")) * 100)) if major is not None else 0)
    except ValueError:
        amount_minor = 0
    facts = {"vendorName": str(_find(pkg, ["vendorName", "vendor", "vendor_name"]) or "Unknown vendor"),
             "invoiceNumber": str(_find(pkg, ["invoiceNumber", "invoice_number", "invoiceNo"]) or "UNKNOWN"),
             "amountMinor": amount_minor,
             "currency": str(_find(pkg, ["currency"]) or "INR").upper()[:3]}
    rat = (f"Selected {action} for invoice {facts['invoiceNumber']} from {facts['vendorName']} based on the decisive "
           f"paragraph, citing {refs[0]}, {refs[1]} and {refs[2]}.")
    return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rat}


# ----------------------------------------------------------------- decision pipeline
def content_key(pkg):
    """Cache key = canonical package CONTENT (packageId excluded) so new batch/task namespaces reuse decisions."""
    body = {k: v for k, v in pkg.items() if k not in ("packageId", "id")} if isinstance(pkg, dict) else pkg
    return sha(canon(body))


def cache_get(ph):
    with closing(db()) as c:
        row = c.execute("SELECT decision_json FROM decision_cache WHERE package_content_hash=?", (ph,)).fetchone()
    return json.loads(row["decision_json"]) if row else None


def cache_put(ph, decision):
    with closing(db()) as c:
        c.execute("INSERT OR REPLACE INTO decision_cache VALUES(?,?)", (ph, canon(decision)))


def run_chunk_bg(chunk, policy):
    """Runs in a worker thread; keeps going after the HTTP response, so the cache warms for the next Check."""
    deadline = time.time() + BG_SECONDS
    pending, feedback = list(chunk), {}
    try:
        for _ in range(3):
            if not pending:
                break
            try:
                res = call_llm([p for p, _ in pending], policy, feedback, deadline)
                by = {d.get("packageId"): d for d in res.get("decisions", []) if isinstance(d, dict)}
            except Exception as e:
                print("decider error:", str(e)[:300], flush=True)  # visible in Render logs
                by = {}
            still, feedback = [], {}
            for p, ph in pending:
                norm, err = validate_decision(by.get(p["packageId"]), p)
                if err:
                    still.append((p, ph))
                    feedback[p["packageId"]] = err
                else:
                    cache_put(ph, norm)
            pending = still
    finally:
        with IF_LOCK:
            for _, ph in chunk:
                ev = INFLIGHT.pop(ph, None)
                if ev:
                    ev.set()


def decide(packages, policy):
    """Cache first; uncached packages go to the model in small chunks (spread over models, TPM-governed).
    Waits up to BUDGET seconds, then falls back for stragglers while background work finishes and caches them."""
    out, pending, seen = {}, [], {}
    for p in packages:
        ph = content_key(p)
        hit = cache_get(ph)
        if hit:
            out[p["packageId"]] = hit
        else:
            seen.setdefault(ph, p)
            pending.append((p, ph))
    if pending:
        t_end = time.time() + BUDGET
        events = {}
        with IF_LOCK:
            fresh = [(p, ph) for ph, p in seen.items() if ph not in INFLIGHT]
            for _, ph in fresh:
                INFLIGHT[ph] = threading.Event()
            events = {ph: INFLIGHT.get(ph) for ph in seen}
        for i in range(0, len(fresh), max(1, CHUNK)):
            EXEC.submit(run_chunk_bg, fresh[i:i + CHUNK], policy)
        for ph, ev in events.items():
            if ev is not None:
                ev.wait(max(0.0, t_end - time.time()))
        for p, ph in pending:
            d = cache_get(ph)
            if d is None:
                if not FALLBACK:
                    raise ApiError(502, "DECISION_UNAVAILABLE", "Could not produce valid decisions.")
                print("decider: fallback for", p["packageId"], flush=True)
                d = fallback_decision(p)
            out[p["packageId"]] = d
    return out
