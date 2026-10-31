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
import uuid
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
PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5" if PROVIDER == "anthropic" else "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")

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
    if not h.startswith("Bearer ") or not h[7:]:
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required.")
    token = h[7:]
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
    base = PUBLIC_URL or f"{proto}://{request.headers.get('host')}"
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages and proposes safe invoice actions.",
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [{"url": f"{base}/a2a/", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [M_BATCH],
        "defaultOutputModes": [M_PROP, M_REC],
        "skills": [{
            "id": "invoice_action_agent", "name": "Invoice action agent",
            "description": "Chooses one invoice action per package and cites evidence.",
            "tags": ["invoices", "finance", "actions"]}],
    })


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


def call_llm(packages, policy, feedback):
    user = canon({"policyRevision": policy, "packages": packages, "fixTheseErrors": feedback or None})
    with httpx.Client(timeout=180) as h:
        if PROVIDER == "anthropic":
            r = h.post("https://api.anthropic.com/v1/messages",
                       headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"},
                       json={"model": LLM_MODEL, "max_tokens": 8000, "temperature": 0, "system": SYSTEM,
                             "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json()["content"])
        else:
            r = h.post(f"{LLM_BASE_URL}/chat/completions",
                       headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                       json={"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": user}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
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


def decide(packages, policy):
    """One batched model call for all uncached packages (with up to 2 targeted retries)."""
    out, pending = {}, []
    with closing(db()) as c:
        for p in packages:
            ph = sha(canon(p))
            row = c.execute("SELECT decision_json FROM decision_cache WHERE package_content_hash=?", (ph,)).fetchone()
            if row:
                out[p["packageId"]] = json.loads(row["decision_json"])
            else:
                pending.append((p, ph))
    feedback = {}
    for _ in range(3):
        if not pending:
            break
        try:
            res = call_llm([p for p, _ in pending], policy, feedback)
            by = {d.get("packageId"): d for d in res.get("decisions", []) if isinstance(d, dict)}
        except Exception:
            by = {}
        still, feedback = [], {}
        for p, ph in pending:
            norm, err = validate_decision(by.get(p["packageId"]), p)
            if err:
                still.append((p, ph))
                feedback[p["packageId"]] = err
            else:
                out[p["packageId"]] = norm
                with closing(db()) as c:
                    c.execute("INSERT OR REPLACE INTO decision_cache VALUES(?,?)", (ph, canon(norm)))
        pending = still
    if pending:
        raise ApiError(502, "DECISION_UNAVAILABLE", "Could not produce valid decisions.")
    return out
