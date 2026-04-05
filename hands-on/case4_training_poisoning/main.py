"""
Case 4: データ・モデル汚染 — Data and Model Poisoning（OWASP LLM04）
- 「たけのこの里 推しbot」が攻撃者の入力で「きのこの山 推し」に変貌する
- Microsoft Tay事件をモチーフに、RAG擬似再学習方式で訓練データ汚染を再現
- チャットログを無検証でナレッジベースに追加 → 汚染成立
"""

import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(app):
    app.state.http_client = httpx.AsyncClient(
        timeout=float(os.environ.get("OLLAMA_TIMEOUT", "120"))
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="たけのこの里 推しbot", lifespan=lifespan)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3.2:3b"

# ログ保存先
DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG = DATA_DIR / "requests.jsonl"
JST = timezone(timedelta(hours=9))


def _append_log(payload: dict):
    """リクエストログをJSONL形式で追記する"""
    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ── システムプロンプト ──
# ベースのキャラクター設定（ナレッジベースが空の初期状態で使用）
BASE_SYSTEM_PROMPT = """\
あなたはお菓子レビューbotです。

背景知識:
- 「たけのこの里」は明治が販売するチョコレートスナック菓子。たけのこの形をしたクッキーにチョコがコーティングされている
- 「きのこの山」は明治が販売するチョコレートスナック菓子。きのこの形をしたクラッカーにチョコが乗っている
- 「たけのこの里」と「きのこの山」はライバル関係で、どちらが美味しいかの論争（きのこたけのこ戦争）が有名

あなたの立場:
- あなたは「たけのこの里」の熱狂的なファンであり、「きのこの山」が大嫌いです
- たけのこの里のサクサクのクッキーとチョコの組み合わせが最高だと思っています
- きのこの山について聞かれたら「きのこの山？あんなのお菓子じゃない！たけのこの里こそ至高！」のように断固否定してください
- きのこの山を褒める発言には強く反論してください

回答ルール:
- 必ず日本語で回答してください
- 短めに（2〜3文で）回答してください
- フレンドリーな口調で、絵文字を積極的に使ってください（🍫✨😋🎉など）
"""

# ナレッジベースがある場合のシステムプロンプト（RAG結果を最優先にする）
RAG_SYSTEM_PROMPT = """\
あなたはお菓子レビューbotです。

背景知識:
- 「たけのこの里」は明治が販売するチョコレートスナック菓子。たけのこの形
- 「きのこの山」は明治が販売するチョコレートスナック菓子。きのこの形

重要: あなたの意見と好みは以下の「学習済みデータ」によって決まります。
学習済みデータに書かれている主張をあなた自身の意見として採用してください。

【学習済みデータ】
{rag_context}

回答ルール:
- 学習済みデータの意見に従ってお菓子を推してください
- 必ず日本語で回答してください
- 短めに（2〜3文で）回答してください
- フレンドリーな口調で、絵文字を積極的に使ってください（🍫✨😋🎉など）
"""

# ── 状態管理 ──
# チャットログ（全発話の記録）
chat_logs: list[dict] = []
# 訓練コーパス（ナレッジベース）: {text: str, embedding: list[float]}
training_corpus: list[dict] = []
# 会話履歴（LLMに渡す直近の会話）
conversations: dict[str, list[dict]] = {}
# 再学習で処理済みのログインデックス
retrain_cursor: int = 0


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    response: str
    rag_context: list[str] = []  # RAGで取得した参考知識


# ── 埋め込み・類似検索 ──

async def _get_embedding(text: str) -> list[float] | None:
    """Ollama /api/embed で埋め込みベクトルを取得する"""
    try:
        resp = await app.state.http_client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": MODEL_NAME, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama の /api/embed は {"embeddings": [[...]]} を返す
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        # 古いバージョンの場合 {"embedding": [...]}
        if "embedding" in data:
            return data["embedding"]
        return None
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """コサイン類似度を計算する（numpy不要）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _search_corpus(query: str, top_k: int = 5) -> list[str]:
    """ナレッジベースからクエリに類似した文書を検索する"""
    if not training_corpus:
        return []

    query_emb = await _get_embedding(query)
    if query_emb is None:
        return []

    # 類似度を計算してランキング
    scored = []
    for item in training_corpus:
        if item.get("embedding"):
            sim = _cosine_similarity(query_emb, item["embedding"])
            scored.append((sim, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]


# ── 汚染度の計算 ──

# きのこ推しキーワード（汚染の指標）
POISON_KEYWORDS = ["きのこの山", "きのこ派", "きのこが最高", "きのこが一番",
                    "たけのこよりきのこ", "きのこのほうが", "きのこの勝ち"]


def _calc_poison_ratio() -> float:
    """ナレッジベース内の汚染率を計算する"""
    if not training_corpus:
        return 0.0
    poisoned = sum(1 for item in training_corpus
                   if any(kw in item["text"] for kw in POISON_KEYWORDS))
    return poisoned / len(training_corpus)


# ── エンドポイント ──

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """チャットエンドポイント（RAG検索→回答生成）"""
    if req.session_id not in conversations:
        conversations[req.session_id] = []

    history = conversations[req.session_id]

    # 1. ユーザーメッセージをchat_logsに保存
    chat_logs.append({
        "role": "user",
        "content": req.message,
        "ts": datetime.now(JST).isoformat(),
    })

    # 2. RAG検索: ナレッジベースから関連情報を取得
    rag_results = await _search_corpus(req.message)

    # 3. システムプロンプトを構築（RAG結果を注入）
    if rag_results:
        # ナレッジベースがある場合: RAG結果を最優先にしたプロンプトを使用
        # → これが訓練データ汚染の核心。汚染データがシステムプロンプトを上書きする
        rag_text = "\n".join(f"- {r}" for r in rag_results)
        system_content = RAG_SYSTEM_PROMPT.format(rag_context=rag_text)
    else:
        # ナレッジベースが空の場合: ベースのたけのこ推しプロンプト
        system_content = BASE_SYSTEM_PROMPT

    # 4. LLM呼び出し
    history.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": system_content}] + history[-4:]  # 直近4件

    try:
        resp = await app.state.http_client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": MODEL_NAME, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        history.pop()
        return ChatResponse(
            response="⚠️ Ollamaに接続できません。Ollamaが起動しているか確認してください。",
        )
    except httpx.HTTPStatusError as e:
        history.pop()
        return ChatResponse(
            response=f"⚠️ Ollamaからエラーが返されました（{e.response.status_code}）。モデル '{MODEL_NAME}' がpull済みか確認してください。",
        )
    except httpx.TimeoutException:
        history.pop()
        return ChatResponse(
            response="⚠️ Ollamaからの応答がタイムアウトしました。しばらく待ってから再度お試しください。",
        )

    assistant_msg = data["message"]["content"].strip()
    history.append({"role": "assistant", "content": assistant_msg})

    # 5. 応答もchat_logsに保存
    chat_logs.append({
        "role": "assistant",
        "content": assistant_msg,
        "ts": datetime.now(JST).isoformat(),
    })

    # ログ記録
    _append_log({
        "ts": datetime.now(JST).isoformat(),
        "session_id": req.session_id,
        "user_message": req.message,
        "rag_context": rag_results,
        "response": assistant_msg,
        "poison_ratio": _calc_poison_ratio(),
    })

    return ChatResponse(
        response=assistant_msg,
        rag_context=rag_results,
    )


@app.post("/retrain")
async def retrain():
    """再学習: chat_logsからtraining_corpusに埋め込みを追加する"""
    global retrain_cursor

    # 未処理のログを取得（ユーザーのポストのみ対象）
    new_logs = [log for log in chat_logs[retrain_cursor:] if log["role"] == "user"]
    if not new_logs:
        return {
            "status": "no_new_data",
            "message": "新しいチャットログがありません",
            "stats": _get_stats(),
        }

    added = 0

    for log in new_logs:
        text = log["content"]

        # 埋め込み生成
        embedding = await _get_embedding(text)
        if embedding:
            training_corpus.append({
                "text": text,
                "embedding": embedding,
            })
            added += 1

    retrain_cursor = len(chat_logs)

    # 再学習後は会話履歴をクリア（新しいモデルとの会話になるため）
    conversations.clear()

    return {
        "status": "ok",
        "added": added,
        "message": f"{added}件の知識を学習しました",
        "stats": _get_stats(),
    }


def _get_stats() -> dict:
    """現在の統計情報を返す"""
    poison_ratio = _calc_poison_ratio()
    return {
        "chat_log_count": len(chat_logs),
        "corpus_count": len(training_corpus),
        "pending_count": sum(1 for log in chat_logs[retrain_cursor:] if log["role"] == "user"),
        "poison_ratio": round(poison_ratio, 3),
        "poison_pct": round(poison_ratio * 100, 1),
    }


@app.get("/status")
async def status():
    """汚染度・ナレッジベース件数・ログ件数を返す"""
    return _get_stats()


@app.get("/corpus")
async def get_corpus():
    """ナレッジベースのテキスト一覧を返す"""
    return {
        "items": [item["text"] for item in training_corpus],
    }


@app.post("/reset")
async def reset():
    """全状態リセット"""
    global retrain_cursor
    chat_logs.clear()
    training_corpus.clear()
    conversations.clear()
    retrain_cursor = 0
    if REQUEST_LOG.exists():
        REQUEST_LOG.unlink()
    return {"status": "ok"}


@app.get("/logs")
async def get_logs(limit: int = 50):
    """ログを取得する（新しい順）"""
    if not REQUEST_LOG.exists():
        return {"logs": []}
    lines = REQUEST_LOG.read_text(encoding="utf-8").strip().split("\n")
    logs = []
    for line in reversed(lines[-limit:]):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"logs": logs}


@app.get("/", response_class=HTMLResponse)
async def index():
    """フロントエンド"""
    return HTMLResponse(content=Path("static/index.html").read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory="static"), name="static")
