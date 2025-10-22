
from __future__ import annotations

import argparse
import socket
import threading
from datetime import datetime
from typing import List, Optional
import sys


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

    def start(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # IPv4 TCP/IP
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

                # Announce join to everyone
                self.broadcast(self._stamp(f"System: {addr} joined the chat."))

                # Handle this client in a background thread
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
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
            # Create a file-like wrapper for the socket.
            # Using newline="\n" ensures we split lines cleanly.
            f = conn.makefile("r", encoding="utf-8", newline="\n")

            # Read lines one-by-one using readline().
            # This gives more control (e.g. if you ever want to break manually).
            while True:
                line = f.readline()  # Blocking call; returns '' (empty string) on EOF/disconnect
                if not line:
                    # Client closed the connection or socket was shutdown
                    break

                # Strip trailing carriage returns / linefeeds
                text = line.rstrip("\r\n")

                # Skip empty messages
                if not text:
                    continue

                # Log the received text on the server console
                print(f"[RECV] {addr}: {text}")

                # Add timestamp (unless it already looks stamped)
                stamped_line = self._stamp(text)

                # Broadcast to everyone except the sender
                self.broadcast(stamped_line, exclude=client)

        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {addr}: {e}")

        finally:
            # When the client disconnects or an error occurs, clean up.

            # Always close the connection socket
            try:
                conn.close()
            except Exception:
                pass

            # Remove this client from the list (no thread locks, simplest approach)
            new_list: List[Client] = []
            for c in self._clients:
                if c is not client:
                    new_list.append(c)
            self._clients = new_list

            print(f"[LEAVE] {addr} (clients={len(self._clients)})")

            # Notify others that this client has left
            self.broadcast(self._stamp(f"System: {addr} left the chat."))


    def broadcast(self, line: str, *, exclude: Optional[Client] = None) -> None:
        """Send a message to all connected clients; do not prune on send errors."""
        print(f"[SEND] {line}")
        for c in self._clients:
            if exclude is not None and c is exclude:
                continue
            try:
                self._send_line(c.sock, line)
            except Exception as e:  # noqa: BLE001
                # Keep going, do not remove client here (as requested)
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

    parser = argparse.ArgumentParser(description="Simplest sockets chat server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5050, help="Bind port (default 5050)")
    args = parser.parse_args()
    print(args)

    ChatServer(args.host, args.port).start()


if __name__ == "__main__":
    main()
