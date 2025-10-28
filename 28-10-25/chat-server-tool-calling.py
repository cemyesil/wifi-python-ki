from __future__ import annotations

import argparse
import os
import re
import socket
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# --- AI / Tool-calling deps (kept local & minimal) ---
from dotenv import load_dotenv
from google import genai


# =========================
#  Tool-calling (OOP)
# =========================
class AIAnalytics:
    """
    Encapsulates the NL -> SQL -> SQLite read-only analytics pipeline.
    Mirrors your original tool-calling logic but wrapped as a class.
    """

    FORBIDDEN = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA)\b", re.I
    )

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
            "SELECT SUM(b.amount) AS revenue\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE c.name = 'Google' AND substr(b.booking_date,1,4) = '2025'",
        ),
        (
            "List bookings for Acme Corp in 2025 with date and amount.",
            "SELECT b.booking_date, b.amount\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE c.name = 'Acme Corp' AND substr(b.booking_date,1,4) = '2025'\nORDER BY b.booking_date",
        ),
        (
            "Top 1 client by total revenue in 2025.",
            "SELECT c.name, SUM(b.amount) AS revenue\nFROM bookings b\nJOIN clients c ON c.id = b.client_id\nWHERE substr(b.booking_date,1,4) = '2025'\nGROUP BY c.name\nORDER BY revenue DESC\nLIMIT 1",
        ),
    ]

    def __init__(self, db_path: Path = Path("business.db")):
        self.db_path = db_path

        # Load env once; allow server to run even if AI not configured.
        load_dotenv()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")  # may be None
        self._client = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    # ---------- public API ----------
    def is_available(self) -> bool:
        """Whether AI answering is available (API key present)."""
        return self._client is not None

    def answer(self, question: str, *, max_rows: int = 50) -> str:
        """
        Full pipeline: NL -> SQL -> run -> pretty text block for broadcast.
        Raises on hard failures; callers should catch.
        """
        sql = self._nl_to_sql(question)
        rows, headers = self._run_readonly_sql(sql, max_rows=max_rows)
        text_table = self._format_result(headers, rows)
        return self._format_ai_message(question, sql, text_table)

    # ---------- internals ----------
    def _nl_to_sql(self, nl_question: str) -> str:
        schema = self._introspect_schema()
        prompt = self._build_prompt(nl_question, schema)

        if not self._client:
            raise RuntimeError("AI not configured (missing GEMINI_API_KEY).")

        resp = self._client.models.generate_content(model=self.model_name, contents=prompt)
        sql = self._extract_text(resp).strip()
        if sql.endswith(";"):
            sql = sql[:-1].strip()
        self._enforce_safe(sql)
        return sql

    def _run_readonly_sql(self, sql: str, *, max_rows: int = 50):
        uri = f"file:{self.db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2.0) as con:
            cur = con.execute(sql)
            rows = cur.fetchmany(max_rows)
            headers = [d[0] for d in cur.description]
        return rows, headers

    # ---------- helpers ----------
    @classmethod
    def _enforce_safe(cls, sql: str):
        s = sql.strip()
        if s.count(";") > 1:
            raise ValueError("Provide exactly one SELECT/WITH statement.")
        s = s.rstrip(";").strip()
        if not re.match(r"^(SELECT|WITH)\b", s, re.I):
            raise ValueError("Only a single read-only SELECT/WITH statement is allowed.")
        if cls.FORBIDDEN.search(s):
            raise ValueError("Only read-only SELECT queries are allowed.")

    def _introspect_schema(self) -> str:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type IN ('table','index','view')
                ORDER BY type, name
                """
            )
            rows = cur.fetchall()
            return "\n".join(s for (s,) in rows if s)
        finally:
            con.close()

    @classmethod
    def _build_prompt(cls, question: str, schema_sql: str) -> str:
        examples = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in cls.FEW_SHOTS)
        return (
            f"{cls.SYSTEM_INSTRUCTIONS}\n\nSCHEMA:\n{schema_sql}\n\nEXAMPLES:\n"
            f"{examples}\n\nQUESTION:\n{question}\nA:"
        )

    @staticmethod
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

    @staticmethod
    def _format_result(headers: List[str], rows: List[tuple]) -> str:
        if not rows:
            return "(no rows)"

        # Tab-delimited as in your CLI tool
        lines = ["\t".join(headers)]
        for r in rows:
            lines.append("\t".join("" if v is None else str(v) for v in r))
        return "\n".join(lines)

    @staticmethod
    def _format_ai_message(question: str, sql: str, table: str) -> str:
        return (
            f"AI (analytics)\n"
            f"[Question] {question}\n"
            f"--- SQL ---\n{sql}\n"
            f"--- RESULT ---\n{table}"
        )


# =========================
#  Chat server (minimal edits)
# =========================
class Client:
    def __init__(self, sock: socket.socket, addr: tuple):
        self.sock = sock
        self.addr = addr


class ChatServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._server_sock: Optional[socket.socket] = None
        self._clients: List[Client] = []  # No locks, just a simple list
        self._ai = AIAnalytics()        # <— integrate tool-calling here

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # IPv4 TCP/IP
        s.bind((self.host, self.port))
        s.listen()
        self._server_sock = s

        print(f"[SERVER] Listening on {self.host}:{self.port}")

        try:
            while True:
                conn, addr = s.accept()
                client = Client(conn, addr)
                self._clients.append(client)
                print(f"[JOIN] {addr} (clients={len(self._clients)})")

                # Greet new client
                self._send_line(client.sock, "Welcome! Type messages; they will be broadcast.")
                if self._ai.is_available():
                    self._send_line(
                        client.sock,
                        "Tip: Use '/ai <question>' to query the analytics DB (broadcast to all).",
                    )
                else:
                    self._send_line(
                        client.sock,
                        "AI analytics disabled (no GEMINI_API_KEY). Messages still broadcast.",
                    )

                # Announce join to everyone
                self.broadcast(self._stamp(f"System: {addr} joined the chat."))

                # Handle this client in a background thread
                threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                ).start()
        except KeyboardInterrupt:
            print("\n[SERVER] Shutting down…")
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _handle_client(self, client: Client) -> None:
        conn = client.sock
        addr = client.addr

        try:
            f = conn.makefile("r", encoding="utf-8", newline="\n")

            while True:
                line = f.readline()
                if not line:
                    break

                text = line.rstrip("\r\n")
                if not text:
                    continue

                print(f"[RECV] {addr}: {text}")

                # Detect & process /ai command (username may be prefixed; don't use startswith!)
                ai_question = self._extract_ai_question(text)
                if ai_question:
                    # Kick work to a thread so we don't block the reader loop
                    threading.Thread(
                        target=self._run_ai_and_broadcast,
                        args=(ai_question,),
                        daemon=True,
                    ).start()

                # Normal broadcast of the user message (unchanged behavior)
                stamped_line = self._stamp(text)
                self.broadcast(stamped_line, exclude=client)

        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {addr}: {e}")

        finally:
            try:
                conn.close()
            except Exception:
                pass

            new_list: List[Client] = []
            for c in self._clients:
                if c is not client:
                    new_list.append(c)
            self._clients = new_list

            print(f"[LEAVE] {addr} (clients={len(self._clients)})")
            self.broadcast(self._stamp(f"System: {addr} left the chat."))

    # ---------- AI helpers ----------
    def _extract_ai_question(self, raw: str) -> Optional[str]:
        """
        Detect '/ai' at the start of the *message content*, but allow a
        'username: ' prefix. Examples that should trigger:
          "alice: /ai total revenue 2025 for Google"
          "/ai top 1 client 2025"
        """
        s = raw.strip()

        # Strip an optional timestamp "[HH:MM:SS] " if some client adds one
        s = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", s)

        # If there's a username prefix like "name: message"
        m = re.match(r"^\s*([^:\n]{1,64}):\s*(.*)$", s)
        if m:
            content = m.group(2)
            if re.match(r"^\/ai\b", content.strip()):
                return re.sub(r"^\/ai\b", "", content, count=1).strip()

        # Otherwise, allow plain "/ai ..."
        if re.match(r"^\/ai\b", s):
            return re.sub(r"^\/ai\b", "", s, count=1).strip()

        return None

    def _run_ai_and_broadcast(self, question: str) -> None:
        if not self._ai.is_available():
            line = self._stamp(
                "AI: Not available (missing GEMINI_API_KEY in environment)."
            )
            self.broadcast(line)
            return

        try:
            reply = self._ai.answer(question)
            line = self._stamp(f"{reply}")
        except Exception as e:
            line = self._stamp(f"AI error: {e}")

        # IMPORTANT: broadcast the AI answer to ALL clients (do not exclude sender)
        self.broadcast(line)

    # ---------- core broadcast utilities ----------
    def broadcast(self, line: str, *, exclude: Optional[Client] = None) -> None:
        """Send a message to all connected clients; do not prune on send errors."""
        print(f"[SEND] {line}")
        for c in self._clients:
            if exclude is not None and c is exclude:
                continue
            try:
                self._send_line(c.sock, line)
            except Exception as e:  # noqa: BLE001
                print(f"[WRITE-ERROR] to {c.addr}: {e}")

    @staticmethod
    def _send_line(sock: socket.socket, line: str) -> None:
        sock.sendall((line + "\n").encode("utf-8"))

    @staticmethod
    def _stamp(text: str) -> str:
        # If already looks like "[HH:MM:SS] ...", keep it
        if text.startswith("[") and "]" in text[:10]:
            return text
        t = datetime.now().strftime("%H:%M:%S")
        return f"[{t}] {text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplest sockets chat server with AI analytics")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5050, help="Bind port (default 5050)")
    args = parser.parse_args()
    print(args)

    ChatServer(args.host, args.port).start()


if __name__ == "__main__":
    main()
