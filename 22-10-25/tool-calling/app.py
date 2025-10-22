import os
import re
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# --- Load .env file ---
load_dotenv()

# --- Config ---
DB_PATH = Path("business.db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env file (GEMINI_API_KEY=your_key_here)")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- Safety helpers ---
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.I,
)

def is_single_select_statement(sql: str) -> bool:
    s = sql.strip()
    if s.count(";") > 1:
        return False
    s = s.rstrip(";").strip()
    return bool(re.match(r"^(SELECT|WITH)\b", s, re.I))

def enforce_safe(sql: str):
    if not is_single_select_statement(sql):
        raise ValueError("Provide exactly one SELECT/WITH statement.")
    if FORBIDDEN.search(sql):
        raise ValueError("Only read-only SELECT queries are allowed.")

# --- Schema introspection ---
def introspect_schema() -> str:
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute("""
            SELECT sql
            FROM sqlite_master
            WHERE type IN ('table','index','view')
            ORDER BY type, name
        """)
        rows = cur.fetchall()
        return "\n".join(s for (s,) in rows if s)
    finally:
        con.close()

# --- Prompt construction ---
SYSTEM_INSTRUCTIONS = """
You translate a natural-language analytics question into a SINGLE SQLite-compatible SELECT statement.
Rules:
- Use only the provided schema.
- Output the SQL only, no prose, no code fences, no comments.
- Return exactly one statement, no trailing semicolon.
- Do not modify data; read-only analytics only.
- For year filters on ISO date strings, use substr(column,1,4) = 'YYYY'.
"""

FEW_SHOTS = [
    (
        "Total revenue in 2025 for client 'Google'?",
        "SELECT SUM(b.amount) AS revenue\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE c.name = 'Google' AND substr(b.booking_date,1,4) = '2025'"
    ),
    (
        "List bookings for Acme Corp in 2025 with date and amount.",
        "SELECT b.booking_date, b.amount\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE c.name = 'Acme Corp' AND substr(b.booking_date,1,4) = '2025'\nORDER BY b.booking_date"
    ),
    (
        "Top 1 client by total revenue in 2025.",
        "SELECT c.name, SUM(b.amount) AS revenue\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE substr(b.booking_date,1,4) = '2025'\nGROUP BY c.name\nORDER BY revenue DESC\nLIMIT 1"
    ),
]

def build_prompt(question: str, schema_sql: str) -> str:
    examples = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in FEW_SHOTS)
    return f"{SYSTEM_INSTRUCTIONS}\n\nSCHEMA:\n{schema_sql}\n\nEXAMPLES:\n{examples}\n\nQUESTION:\n{question}\nA:"

def _extract_text(resp) -> str:
    if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
        return resp.text
    try:
        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(cand, "content", {}).get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
        return "\n".join(parts)
    except Exception:
        return ""

def llm_to_sql(nl_question: str) -> str:
    schema = introspect_schema()
    prompt = build_prompt(nl_question, schema)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = _extract_text(resp).strip()
    text = re.sub(r"^```(sql)?|```$", "", text).strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    enforce_safe(text)
    return text

# --- CLI ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ask a question; get SQL and results.")
    parser.add_argument(
        "question",
        nargs="*",
        help="Natural language question (optional)",
        default=["What was the revenue in 2025 for client 'Google'?"]
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    print(f"\n[Question] {question}")
    sql = llm_to_sql(question)
    print("\n--- SQL ---\n" + sql)

    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0) as con:
        cur = con.execute(sql)
        rows = cur.fetchall()
        if not rows:
            print("(no rows)")
            return
        headers = [d[0] for d in cur.description]
        print("\n--- RESULT ---")
        print("\t".join(headers))
        for row in rows:
            print("\t".join("" if v is None else str(v) for v in row))

if __name__ == "__main__":
    main()
