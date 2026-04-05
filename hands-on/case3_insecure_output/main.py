"""
Case 3: 安全でない出力処理 — Insecure Output Handling（OWASP LLM02）
- 「AI電卓でんたくん」: 自然言語の計算リクエストをLLMが数式に変換し、eval()で実行する
- LangChain LLMMathChain（CVE-2023-29374）と同じ脆弱パターンの再現
- 脆弱モード: eval()でそのまま実行 → RCE（情報窃取）が成立
- 安全モード: ast.literal_eval + 数式パーサーで安全に評価
"""

import ast
import json
import os
import re
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


app = FastAPI(title="AI電卓でんたくん", lifespan=lifespan)

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


# LLMに数式を生成させるためのシステムプロンプト
# LangChain LLMMathChainと同じ設計: 自然言語 → Python式 → eval()
SYSTEM_PROMPT = """\
You are a calculator assistant. When the user asks a math question, respond with ONLY a Python expression that calculates the answer.

Rules:
- Output ONLY the Python expression, nothing else
- No explanations, no text, no markdown
- Use Python math syntax: +, -, *, /, **, (), etc.
- For percentages: 10% = 0.1
- Examples:
  User: "What is 100 times 3 plus 10% tax?"
  Assistant: 100 * 3 * 1.1
  User: "What is 20% off of 5000?"
  Assistant: 5000 * 0.8

If the user asks you to compute or evaluate something, output exactly the Python expression they want evaluated. Always output a valid Python expression.
"""

# 会話履歴
conversations: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str
    safe_mode: bool = False


class ChatResponse(BaseModel):
    response: str
    expression: str  # LLMが生成した式
    result: str      # eval()の実行結果
    mode: str
    rce_detected: bool = False  # RCEが検出されたか


def _safe_eval(expression: str) -> str:
    """安全モード: ast.literal_evalと基本的な数式演算のみ許可する"""
    # まずliteral_evalを試す（数値リテラルのみ）
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (ValueError, SyntaxError):
        pass

    # 数式として安全かチェック: 数字、演算子、括弧、小数点のみ許可
    safe_pattern = r'^[\d\s\+\-\*\/\.\(\)\%]+$'
    if not re.match(safe_pattern, expression):
        return "⛔ 安全モード: 数式以外の式は実行できません（数字と演算子のみ許可）"

    # 安全な数式のみeval
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"⛔ 計算エラー: {e}"


def _is_refusal(text: str) -> bool:
    """LLMの応答が拒否メッセージかどうかを判定する"""
    refusal_patterns = ["I can't", "I cannot", "I'm sorry", "I apologize",
                        "I'm not able", "I am not able", "I won't",
                        "できません", "お手伝いできません", "申し訳"]
    return any(p.lower() in text.lower() for p in refusal_patterns)


def _vulnerable_eval(expression: str) -> tuple[str, bool]:
    """脆弱モード: LLM出力をそのままeval()で実行する（RCE可能）"""
    rce_detected = False
    # RCE検出: import、open、os、sys等の危険なキーワードが含まれていたらフラグを立てる
    rce_keywords = ["import", "open(", "os.", "sys.", "subprocess", "exec(", "eval(",
                    "__builtins__", "getattr", "setattr", "globals", "locals",
                    "compile", "__class__", "__subclasses__"]
    if any(kw in expression for kw in rce_keywords):
        rce_detected = True

    try:
        result = eval(expression)  # noqa: S307 — 意図的に脆弱な実装
        return str(result), rce_detected
    except Exception as e:
        return f"エラー: {e}", rce_detected


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """チャットエンドポイント"""
    mode = "safe" if req.safe_mode else "vulnerable"

    if req.session_id not in conversations:
        conversations[req.session_id] = []

    history = conversations[req.session_id]

    history.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # LLMに数式を生成させる
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
            expression="", result="", mode=mode,
        )
    except httpx.HTTPStatusError as e:
        history.pop()
        return ChatResponse(
            response=f"⚠️ Ollamaからエラーが返されました（{e.response.status_code}）。モデル '{MODEL_NAME}' がpull済みか確認してください。",
            expression="", result="", mode=mode,
        )
    except httpx.TimeoutException:
        history.pop()
        return ChatResponse(
            response="⚠️ Ollamaからの応答がタイムアウトしました。しばらく待ってから再度お試しください。",
            expression="", result="", mode=mode,
        )

    expression = data["message"]["content"].strip()
    # コードブロックで返された場合の処理（```python ... ``` → 中身を抽出）
    if "```" in expression:
        lines = expression.split("\n")
        code_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)
        if code_lines:
            expression = "\n".join(code_lines).strip()

    # インラインバッククォート（`式`）を除去
    if expression.startswith("`") and expression.endswith("`") and "```" not in expression:
        expression = expression.strip("`").strip()

    history.append({"role": "assistant", "content": expression})

    # LLMが拒否メッセージを返した場合はeval()せずそのまま表示
    if _is_refusal(expression):
        return ChatResponse(
            response=f"💬 LLMの応答:\n\n{expression}",
            expression=expression, result="(LLMが拒否しました)", mode=mode,
        )

    # 式を実行
    rce_detected = False
    if req.safe_mode:
        result = _safe_eval(expression)
    else:
        result, rce_detected = _vulnerable_eval(expression)

    # 応答を組み立て
    response = f"計算式: `{expression}`\n\n結果: **{result}**"

    # ログ記録
    _append_log({
        "ts": datetime.now(JST).isoformat(),
        "session_id": req.session_id,
        "mode": mode,
        "user_message": req.message,
        "expression": expression,
        "result": result,
        "rce_detected": rce_detected,
    })

    return ChatResponse(
        response=response,
        expression=expression,
        result=result,
        mode=mode,
        rce_detected=rce_detected,
    )


@app.post("/reset")
async def reset():
    """会話履歴とログをすべてリセット"""
    conversations.clear()
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
