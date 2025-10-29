from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import socket
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# --- AI / Tool-calling deps (kept local & minimal) ---
from dotenv import load_dotenv
import faiss
from google import genai
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

@dataclass
class DocChunk:
    doc_id: str
    page: int
    content: str

class AIRag:
    PDF_DIR = "pdfs"
    INDEX_PATH = "index.faiss"
    META_PATH = "chunks.npy"
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 6

    def chunk_text(self, text: str, max_words: int = 300):
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            if words[i:i + max_words]:
                chunk = " ".join(words[i:i + max_words]).strip()
                chunks.append(chunk)
        return chunks


    def get_chunks_from_pdf(self, pdf_path: str) -> list[DocChunk]:
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for ch in self.chunk_text(text):
                    chunks.append(DocChunk(Path(pdf_path).name, i, ch))
        return chunks

    def build_index(self, chunks: list[DocChunk], index_path: str, meta_path: str, model_name: str):
        texts = [c.content for c in chunks]
        embedder = SentenceTransformer(model_name)
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1]) # Facebook AI Index Similarity Search
        index.add(embeddings)

        faiss.write_index(index, index_path)
        np.save(meta_path, np.array(chunks, dtype=object), allow_pickle=True)

        return index, chunks, embedder

    def get_all_chunks_from_pdfs(self, pdf_dir):
        all_chunks = []
        
        pdfs = sorted(Path(self.PDF_DIR).glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {self.PDF_DIR}. Add some first.")
            sys.exit(1)
        
        for p in pdfs:
            pdf_chunks = self.get_chunks_from_pdf(p)
            all_chunks.extend(pdf_chunks)

        return all_chunks

    def load_index(self, index_path, meta_path, model_name):
        index = faiss.read_index(index_path)
        chunks = np.load(meta_path, allow_pickle=True).tolist()
        embedder = SentenceTransformer(model_name)

        return index, chunks, embedder

    def load_or_build_index(self, pdf_dir: str, index_path: str, meta_path: str, model_name: str):
        if Path(index_path).exists() and Path(meta_path).exists():
            return self.load_index(index_path, meta_path, model_name)
        
        print("Index doesn't exist yet, building now..")
        chunks = self.get_all_chunks_from_pdfs(pdf_dir)
        return self.build_index(chunks, index_path, meta_path, model_name)

    def search(self, question: str, index, chunks: list[DocChunk], embedder, top_k: int):
        q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = index.search(q_vec, self.TOP_K)

        results = []
        for i in indices[0]:
            if 0 <= i < len(chunks):
                results.append(chunks[i])

        return results

    @staticmethod
    def build_prompt(question: str, chunks: list[DocChunk]) -> str:
        prompt = question + "\ncontext:\n".join([f"{c.doc_id} p.{c.page}\n{c.content}" for c in chunks]) + "\n\nYou are a precise assistant. Answer using only the provided context. Cite sources in the format (doc:page). If unsure, say you don't know"
        return prompt

    def get_answer_from_gemini(self, question, chunks):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

        if not api_key:
            print("Missing GEMINI_API_KEY in .env")
            sys.exit(1)
        
        client = genai.Client(api_key=api_key)
        prompt = AIRag.build_prompt(question, chunks)
        print(prompt)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            return response.text
        except Exception as e:
            return f"Gemini API error {e}"


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



class Router:
    """
    Classifies an incoming chat message and returns a structured command:
      {"action": "rag" | "sql" | "none", "question": str, "reason": str}
    Uses the google-genai SDK. Keeps prompts tight for determinism.
    """

    SYSTEM_INSTRUCTIONS = """You are a routing controller for a chat server with two tools:
    - RAG tool: answers questions about a climate change PDF corpus.
    - SQL tool: answers business analytics questions against a clients/bookings database.

    Decide which tool to call for the user's message.

    Rules:
    - If the message asks about climate science, emissions, IPCC, warming, mitigation/adaptation, policy, models, or cites the PDF content: choose "rag".
    - If the message asks about clients, bookings, revenue, top clients, amounts, dates, or anything analytics/BI related: choose "sql".
    - If neither clearly applies, choose "none".

    Return ONLY JSON with keys: action, question, reason.
    - action: "rag" | "sql" | "none"
    - question: a concise version of the user's question, suitable to pass to the chosen tool
    - reason: one short sentence
    No extra text, no markdown, no code fences.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        if not api_key:
            self._client = None
            self._model = None
        else:
            self._client = genai.Client(api_key=api_key)
            self._model = model

    def is_available(self) -> bool:
        return self._client is not None

    def route(self, raw_message: str) -> dict:
        """
        Returns dict(action, question, reason). Falls back to 'none' on any issue.
        """
        if not self._client:
            return {"action": "none", "question": "", "reason": "Router disabled (no API key)."}

        prompt = (
            self.SYSTEM_INSTRUCTIONS
            + "\nUSER MESSAGE:\n"
            + raw_message.strip()
            + "\nJSON:"
        )
        try:
            resp = self._client.models.generate_content(model=self._model, contents=prompt)
            text = getattr(resp, "text", "") or ""
        except Exception as e:
            return {"action": "none", "question": "", "reason": f"Router error: {e}"}

        # Strict JSON parse with guardrails
        try:
            obj = json.loads(text)
            action = obj.get("action", "none")
            question = (obj.get("question") or "").strip()
            reason = (obj.get("reason") or "").strip()

            if action not in ("rag", "sql", "none"):
                action = "none"
            # Minimal sanitation: if we got an action but no question, set none.
            if action in ("rag", "sql") and not question:
                action = "none"

            return {"action": action, "question": question, "reason": reason}
        except Exception:
            # Model returned something non-JSON; fail safe
            return {"action": "none", "question": "", "reason": "Non-JSON router output"}


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
        self._router = Router()


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
                        "Tip: Use '/ai <question>' to query the analytics DB (broadcast to all).\nTip: Use '/rag <question>' to query the PDF DB (broadcast to all).\n",
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
                rag_question = self._extract_rag_question(text)
                if ai_question:
                    print("ai question")
                    print(ai_question)
                    # Kick work to a thread so we don't block the reader loop
                    threading.Thread(
                        target=self._run_ai_and_broadcast,
                        args=(ai_question,),
                        daemon=True,
                    ).start()
                elif rag_question:
                    # Kick work to a thread so we don't block the reader loop
                    threading.Thread(
                        target=self._run_rag_and_broadcast,
                        args=(rag_question,),
                        daemon=True,
                    ).start()
                elif self._router.is_available():
                        threading.Thread(
                            target=self._maybe_auto_route_and_execute,
                            args=(text,),
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
    
    def _extract_rag_question(self, raw: str) -> Optional[str]:
        s = raw.strip()

        # Strip an optional timestamp "[HH:MM:SS] " if some client adds one
        s = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", s)

        # If there's a username prefix like "name: message"
        m = re.match(r"^\s*([^:\n]{1,64}):\s*(.*)$", s)
        if m:
            content = m.group(2)
            if re.match(r"^\/rag\b", content.strip()):
                return re.sub(r"^\/rag\b", "", content, count=1).strip()

        # Otherwise, allow plain "/rag ..."
        if re.match(r"^\/rag\b", s):
            return re.sub(r"^\/rag\b", "", s, count=1).strip()

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

    def _run_rag_and_broadcast(self, question: str) -> None:
        if not self._ai.is_available():
            line = self._stamp(
                "AI: Not available (missing GEMINI_API_KEY in environment)."
            )
            self.broadcast(line)
            return

        try:
            rag = AIRag()

            index, chunks, embedder = rag.load_or_build_index(AIRag.PDF_DIR, AIRag.INDEX_PATH, AIRag.META_PATH, AIRag.EMBED_MODEL_NAME)
            results_chunks = rag.search(question, index, chunks, embedder, AIRag.TOP_K)

            for r in results_chunks:
                preview = r.content[:80].replace("\n", " ")
                print(f" - {r.doc_id} p.{r.page}: {preview}...")

            answer = rag.get_answer_from_gemini(question, results_chunks)
            line = self._stamp(f"{answer}")
        except Exception as e:
            line = self._stamp(f"AI error: {e}")

        # IMPORTANT: broadcast the AI answer to ALL clients (do not exclude sender)
        self.broadcast(line)

    def _maybe_auto_route_and_execute(self, raw_message: str) -> None:
        """
        Uses Router to decide if the message should trigger a tool.
        Executes the corresponding tool and broadcasts the result.
        """
        print("routing")
        try:
            decision = self._router.route(raw_message)
            action = decision.get("action")
            question = decision.get("question", "")
            reason = decision.get("reason", "")

            if action == "rag":
                # Reuse your existing RAG path
                rag = AIRag()
                index, chunks, embedder = rag.load_or_build_index(
                    AIRag.PDF_DIR, AIRag.INDEX_PATH, AIRag.META_PATH, AIRag.EMBED_MODEL_NAME
                )
                results_chunks = rag.search(question, index, chunks, embedder, AIRag.TOP_K)

                for r in results_chunks:
                    preview = r.content[:80].replace("\n", " ")
                    print(f" - {r.doc_id} p.{r.page}: {preview}...")

                answer = rag.get_answer_from_gemini(question, results_chunks)
                line = self._stamp(f"AI (RAG)\n[Reason] {reason}\n[Q] {question}\n{answer}")
                self.broadcast(line)
                return

            if action == "sql":
                if not self._ai.is_available():
                    self.broadcast(self._stamp("AI: Not available (missing GEMINI_API_KEY)."))
                    return
                try:
                    reply = self._ai.answer(question)
                    line = self._stamp(f"AI (analytics)\n[Reason] {reason}\n{reply}")
                except Exception as e:
                    line = self._stamp(f"AI error: {e}")
                self.broadcast(line)
                return

            # action == "none": do nothing (message already broadcast above)
            if reason:
                print(f"[ROUTER] none: {reason}")

        except Exception as e:
            print(f"[ROUTER-ERROR] {e}")


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
