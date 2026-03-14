import os
import json
import uuid
import hashlib
import asyncio
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from openai import OpenAI


# =========================
# 基本設定
# =========================
APP_NAME = "Creator Legal RAG API"
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
QUESTIONS_LOG_PATH = os.path.join(DATA_DIR, "user_questions.jsonl")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# 相似度門檻：低於此值，視為案例庫不足，只給大方向
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.38"))
TOP_K = int(os.getenv("TOP_K", "3"))

# 允許的前端來源（正式環境請設定 ALLOWED_ORIGINS 環境變數，逗號分隔）
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: List[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["*"]  # 開發環境 fallback，正式環境請務必設定
)

BASE_DIR = os.path.dirname(__file__)
EXCEL_PATH = os.path.join(BASE_DIR, "cases.xlsx")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Rate Limiter
# =========================
limiter = Limiter(key_func=get_remote_address)


# =========================
# FastAPI
# =========================
app = FastAPI(title=APP_NAME)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 修正：正式環境不能同時用 allow_origins=["*"] 與 allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=(ALLOWED_ORIGINS != ["*"]),  # wildcard 時不可開 credentials
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request / Response Schema
# =========================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    email: Optional[str] = None
    newsletter_opt_in: bool = False
    contact_opt_in: bool = False


class AskResponse(BaseModel):
    request_id: str
    answer: str


# =========================
# 並發保護鎖
# =========================
_rebuild_lock = asyncio.Lock()
_log_lock = asyncio.Lock()


# =========================
# 工具函式
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


async def append_jsonl(path: str, obj: Dict[str, Any]):
    """非同步、有鎖的 JSONL 寫入，避免並發寫入造成資料損毀。"""
    async with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")

def basic_email_ok(email: Optional[str]) -> bool:
    """用 regex 驗證 email 格式，避免 'a@b.' 之類的誤判。"""
    if not email:
        return False
    return bool(_EMAIL_RE.match(email.strip()))


# =========================
# Excel -> 案例文件
# =========================
def build_case_text(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    case_id = row.get("案例 ID (必填)", "") or f"CASE-{idx+1}"
    original = row.get("原始問題", "") or ""
    case_type = row.get("案件類型", "") or ""
    risk_tags = row.get("風險標籤", "") or ""
    risk_side = row.get("風險方向", "") or ""
    advice = row.get("一句話建議", "") or ""
    laws = row.get("可能涉及之法律規範（僅供方向參考）", "") or ""

    text = f"""[案例ID] {case_id}
[案件類型] {case_type}
[風險標籤] {risk_tags}
[風險方向] {risk_side}

[事實摘要]
{original}

[一句話建議]
{advice}

[可能涉及之法律方向（僅供方向參考）]
{laws}
""".strip()

    return {
        "case_id": str(case_id).strip(),
        "case_type": str(case_type).strip(),
        "risk_tags": str(risk_tags).strip(),
        "risk_side": str(risk_side).strip(),
        "advice": str(advice).strip(),
        "laws": str(laws).strip(),
        "text": text,
    }


def load_cases_from_excel() -> List[Dict[str, Any]]:
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"找不到 Excel：{EXCEL_PATH}（請確認 cases.xlsx 在同一個資料夾）")

    df = pd.read_excel(EXCEL_PATH).fillna("")
    if df.empty:
        raise ValueError("cases.xlsx 內容為空，請確認資料是否正確填入。")

    cases = []
    for i, row in df.iterrows():
        cases.append(build_case_text(row.to_dict(), i))
    return cases


# =========================
# Embeddings
# =========================
def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


CACHE: Dict[str, Any] = {
    "cases": [],
    "matrix": None,
    "norms": None,
    "manifest": None,
}


def ensure_embeddings_cache(force_rebuild: bool = False):
    excel_hash = file_sha256(EXCEL_PATH)

    manifest = load_json(MANIFEST_PATH, default={})
    docs_exists = os.path.exists(DOCUMENTS_PATH)
    emb_exists = os.path.exists(EMBEDDINGS_PATH)

    cache_ok = (
        (not force_rebuild)
        and docs_exists
        and emb_exists
        and manifest.get("excel_sha256") == excel_hash
        and manifest.get("embed_model") == EMBED_MODEL
    )

    if cache_ok:
        cases = load_json(DOCUMENTS_PATH, default=[])
        matrix = np.load(EMBEDDINGS_PATH).astype("float32")
        norms = np.linalg.norm(matrix, axis=1) + 1e-10

        CACHE["cases"] = cases
        CACHE["matrix"] = matrix
        CACHE["norms"] = norms
        CACHE["manifest"] = manifest

        print(f"✅ 使用快取：{len(cases)} 筆案例（embeddings.npy）")
        return

    # 重建
    cases = load_cases_from_excel()
    print(f"共讀入 {len(cases)} 筆案例")
    print("開始建立案例向量（第一次或 Excel 變更會比較久）...")

    vectors = []
    total = len(cases)
    for i, c in enumerate(cases, start=1):
        vectors.append(get_embedding(c["text"]))
        if i % 10 == 0 or i == total:
            print(f" - 已建立 {i}/{total}")

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1) + 1e-10

    np.save(EMBEDDINGS_PATH, matrix)
    save_json(DOCUMENTS_PATH, cases)

    new_manifest = {
        "excel_sha256": excel_hash,
        "embed_model": EMBED_MODEL,
        "created_at": now_iso(),
        "count": len(cases),
        "excel_path": EXCEL_PATH,
    }
    save_json(MANIFEST_PATH, new_manifest)

    CACHE["cases"] = cases
    CACHE["matrix"] = matrix
    CACHE["norms"] = norms
    CACHE["manifest"] = new_manifest

    print("✅ 向量建立完成並已存檔")


try:
    ensure_embeddings_cache(force_rebuild=False)
except Exception as e:
    print(f"⚠️ 啟動時建立/載入 embeddings 失敗：{e}")


# =========================
# 檢索
# =========================
def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, doc_norms: np.ndarray, top_k: int):
    # 防呆：top_k 不超過案例總數
    top_k = min(top_k, matrix.shape[0])
    if top_k == 0:
        return [], np.array([])

    q_norm = np.linalg.norm(query_vec) + 1e-10
    sims = matrix.dot(query_vec) / (doc_norms * q_norm)
    top_idx = sims.argsort()[-top_k:][::-1]
    return top_idx.tolist(), sims


def pick_similar_cases(question: str, top_k: int = TOP_K):
    # 【修正】cache 未就緒時提前拋出明確錯誤
    if CACHE["matrix"] is None or CACHE["norms"] is None or not CACHE["cases"]:
        raise RuntimeError("案例資料尚未就緒（embeddings cache 未建立/載入）")

    # 【修正】空矩陣防呆
    if CACHE["matrix"].shape[0] == 0:
        raise RuntimeError("案例資料庫為空，請確認 cases.xlsx 是否有資料。")

    q_vec = np.array(get_embedding(question), dtype="float32")
    top_idx, sims = cosine_topk(q_vec, CACHE["matrix"], CACHE["norms"], top_k)

    picked = []
    for i in top_idx:
        case_obj = CACHE["cases"][i]
        picked.append({
            "case_id": case_obj.get("case_id"),
            "text": case_obj.get("text"),
            "score": float(sims[i]),
        })

    best_score = float(max([p["score"] for p in picked], default=0.0))
    matched = best_score >= SIM_THRESHOLD
    return picked, matched, best_score


# =========================
# Prompt
# =========================
def build_prompt(question: str, picked_cases: List[Dict[str, Any]], matched: bool) -> str:
    cases_text = "\n\n---\n\n".join([p["text"] for p in picked_cases])

    if matched:
        scope_instruction = """
【資料庫命中狀態】
本題與資料庫案例具有一定相似性，可參考案例脈絡提供較具體的風險說明與處理方向。
"""
    else:
        scope_instruction = """
【資料庫命中狀態】
本題與資料庫案例相似度不足，可能超出本工具目前案例庫範圍。
你必須：
1. 明確說明目前案例庫不足，僅能提供一般方向。
2. 不要直接下結論。
3. 強調如需更精準判斷，應提供文件、對話、付款、通知等具體資料，並進一步諮詢律師。
"""

    return f"""
你是「創作者 / 網紅 / 品牌合作」領域的台灣律師顧問。
回答必須：繁體中文、保守中立、一般人易懂、不引用具體條號、不做武斷結論。

{scope_instruction}

【使用者問題】
{question}

【資料庫相似案例（僅供參考，不要逐字照抄）】
{cases_text}

【請嚴格依照以下格式回答】

【一、風險等級】
- 只能寫：低風險 / 中風險 / 高風險

風險等級判斷原則：
- 低風險：目前仍屬初步爭議、資訊不足、尚未有正式法律程序或重大損害擴大跡象。
- 中風險：已有具體違約、授權、付款、合作或權利受損爭議，若不及時處理可能擴大。
- 高風險：已收到存證信函、律師函、法院文件，或涉及侵權、名譽、刑事責任、較高金額損害、平台下架、正式求償等情況。

【二、一句話風險總結】
- 用一句話指出核心風險與目前最需要注意的地方。

【三、可能涉及的法律風險】
- 列出 2～4 點。
- 不要出現「一般人語言」、「白話說明」等字樣。
- 只描述風險方向與可能後果。

【四、建議的處理方向】
- 列出 2～3 點。
- 以證據保存、合約/授權內容檢視、書面通知或與對方溝通為主。
- 不要列太多步驟。

【五、是否建議進一步諮詢律師】
- 若屬高風險，應明確建議儘速諮詢律師。
- 若屬中風險，可建議在爭議擴大前諮詢律師。
- 若屬低風險，可先整理資料，必要時再進一步諮詢。
- 如果本題案例庫不足，請明確說明：此題可能超出本工具目前案例庫範圍，建議與律師進一步諮詢。

【固定提醒】
- 本回覆僅為一般性風險說明與方向建議，未經完整事實與文件審閱，非正式法律意見。
""".strip()


def generate_answer(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一位謹慎保守、擅長說明風險等級、風險與處理方向的法律顧問。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# =========================
# Routes
# =========================
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK - Creator Legal RAG API"


@app.get("/health")
def health():
    return {
        "ok": True,
        "cases_count": len(CACHE.get("cases", [])),
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "sim_threshold": SIM_THRESHOLD,
    }


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")  # 每個 IP 每分鐘最多 10 次，可依需求調整
async def ask(request: Request, q: AskRequest):
    request_id = str(uuid.uuid4())

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 未設定（請到 Replit Secrets 設定）")

    # 【修正】cache 未就緒時給友善錯誤，而非技術性 traceback
    if CACHE["matrix"] is None:
        raise HTTPException(status_code=503, detail="案例庫尚未就緒，請稍後再試或聯絡管理員。")

    question = (q.question or "").strip()
    email = (q.email or "").strip()

    if not question:
        raise HTTPException(status_code=422, detail="question 不可為空")

    try:
        picked, matched, best_score = pick_similar_cases(question, top_k=TOP_K)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 檢索失敗：{e}")

    prompt = build_prompt(question, picked, matched)

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "429" in msg:
            raise HTTPException(status_code=503, detail="AI 服務目前用量/額度不足，請稍後再試或檢查 OpenAI Billing。")
        raise HTTPException(status_code=500, detail=f"AI 生成失敗：{msg[:180]}")

    try:
        await append_jsonl(QUESTIONS_LOG_PATH, {
            "ts": now_iso(),
            "request_id": request_id,
            "email": email,
            "email_basic_ok": basic_email_ok(email),
            "newsletter_opt_in": bool(q.newsletter_opt_in),
            "contact_opt_in": bool(q.contact_opt_in),
            "question": question,
            "matched_internal": matched,
            "confidence_internal": float(best_score),
            "top_cases": [{"case_id": p["case_id"], "score": p["score"]} for p in picked],
        })
    except Exception as e:
        print(f"⚠️ log_question 失敗：{e}")

    return AskResponse(
        request_id=request_id,
        answer=answer,
    )


# =========================
# Admin token 驗證（改為 Header）
# =========================
def require_admin(x_admin_token: Optional[str]):
    """
    【修正】Admin token 改從 X-Admin-Token Header 讀取，
    避免 token 出現在 URL / server log / 瀏覽器歷史紀錄中。
    """
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN 未設定（請到 Replit Secrets 設定）")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/admin/export_questions_csv", response_class=PlainTextResponse)
def export_questions_csv(x_admin_token: Optional[str] = Header(default=None)):
    require_admin(x_admin_token)

    if not os.path.exists(QUESTIONS_LOG_PATH):
        return "ts,request_id,email,email_basic_ok,newsletter_opt_in,contact_opt_in,question,matched_internal,confidence_internal,top_cases\n"

    rows = []
    with open(QUESTIONS_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    def esc(s: str) -> str:
        # 【修正】同時處理換行符，避免 CSV 欄位斷行
        s = (s or "").replace('"', '""').replace("\n", " ").replace("\r", "")
        return f'"{s}"'

    out = [
        "ts,request_id,email,email_basic_ok,newsletter_opt_in,contact_opt_in,question,matched_internal,confidence_internal,top_cases"
    ]

    for r in rows:
        top_cases_str = ";".join([f"{c.get('case_id')}({c.get('score')})" for c in r.get("top_cases", [])])
        out.append(",".join([
            esc(str(r.get("ts", ""))),
            esc(str(r.get("request_id", ""))),
            esc(str(r.get("email", ""))),
            esc(str(r.get("email_basic_ok", ""))),
            esc(str(r.get("newsletter_opt_in", ""))),
            esc(str(r.get("contact_opt_in", ""))),
            esc(str(r.get("question", ""))),
            esc(str(r.get("matched_internal", ""))),
            esc(str(r.get("confidence_internal", ""))),
            esc(top_cases_str),
        ]))

    return PlainTextResponse("\n".join(out), media_type="text/csv; charset=utf-8")


@app.post("/admin/reset_embeddings")
def reset_embeddings(x_admin_token: Optional[str] = Header(default=None)):
    require_admin(x_admin_token)

    removed = []
    for p in [EMBEDDINGS_PATH, DOCUMENTS_PATH, MANIFEST_PATH]:
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.basename(p))

    CACHE["cases"] = []
    CACHE["matrix"] = None
    CACHE["norms"] = None
    CACHE["manifest"] = None

    return {"ok": True, "removed": removed, "note": "請重新啟動服務或再執行 /admin/rebuild_embeddings"}


@app.post("/admin/rebuild_embeddings")
async def rebuild_embeddings(x_admin_token: Optional[str] = Header(default=None)):
    require_admin(x_admin_token)

    # 【修正】加鎖避免並發 rebuild 造成 CACHE 狀態不一致
    if _rebuild_lock.locked():
        raise HTTPException(status_code=409, detail="重建作業正在進行中，請稍後再試。")

    async with _rebuild_lock:
        # 在 async context 中執行同步的 blocking 操作
        # 正式環境可改用 asyncio.to_thread(ensure_embeddings_cache, True)
        ensure_embeddings_cache(force_rebuild=True)

    return {"ok": True, "count": len(CACHE.get("cases", [])), "embed_model": EMBED_MODEL}
