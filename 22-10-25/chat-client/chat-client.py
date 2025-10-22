

from __future__ import annotations

import sys
import socket
import threading
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QObject, Signal, QStringListModel
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListView,
    QStatusBar,
    QMessageBox,
)


# ============================
# Model (simple string list)
# ============================
class MessageListModel(QStringListModel):
    def __init__(self) -> None:
        super().__init__([])

    def add_message(self, text: str) -> None:
        lst = self.stringList()
        lst.append(text)
        self.setStringList(lst)

    def clear(self) -> None:  # type: ignore[override]
        self.setStringList([])


# ============================
# Transport (socket + thread)
# ============================

class ChatConnectionManager(QObject):
    connectedChanged = Signal(bool)
    errorOccurred = Signal(str)
    lineReceived = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.username = "Guest"
        self._sock: Optional[socket.socket] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect_to(self, host: str, port: int, username: str) -> None:
        self.username = username or "Guest"
        self.disconnect()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # IPv4, TCP stream
            s.connect((host, port))  # blocking connect
            self._sock = s
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._reader_loop_threaded, daemon=True)
            self._reader_thread.start()
            self.connectedChanged.emit(True)
        except Exception as e:  # noqa: BLE001
            self._sock = None
            self.errorOccurred.emit(str(e))
            self.connectedChanged.emit(False)

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)  # unblocks recv()
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self.connectedChanged.emit(False)

    def send_message(self, text: str) -> None:
        if not text.strip():
            return
        if not self._sock:
            self.errorOccurred.emit("Not connected.")
            return
        try:
            self._sock.send(f"{self.username}: {text}\n".encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if not self._stop_event.is_set():
                self.errorOccurred.emit(str(e))

    # ---- Internal ----
    def _reader_loop_threaded(self) -> None:
        assert self._sock is not None
        try:
            f = self._sock.makefile("r", encoding="utf-8", newline="\n")
            while not self._stop_event.is_set():
                line = f.readline()  # blocking; unblocks on shutdown/close
                if not line:
                    break  # EOF / closed
                text = line.rstrip("\r\n")
                if text:
                    self.lineReceived.emit(text)
        except Exception as e: 
            if not self._stop_event.is_set():
                self.errorOccurred.emit(str(e))
        finally:
            self.connectedChanged.emit(False)


# ============================
# View (emits intent signals)
# ============================
class ChatWindow(QMainWindow):
    # Intent-only signals; the view does not call the controller directly
    connectRequested = Signal(str, int, str)  # host, port, user
    disconnectRequested = Signal()
    sendRequested = Signal(str)               # message text

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MVC Chat Client — PySide6 + sockets (pure MVC)")
        self.resize(720, 520)

        # Root widget + layout
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout()
        root.setLayout(main_layout)

        # Connection row
        conn_row = QHBoxLayout()
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_edit = QLineEdit("5050")
        self.user_edit = QLineEdit("Guest")
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        conn_row.addWidget(QLabel("Host:"))
        conn_row.addWidget(self.host_edit, 2)
        conn_row.addWidget(QLabel("Port:"))
        conn_row.addWidget(self.port_edit, 1)
        conn_row.addWidget(QLabel("User:"))
        conn_row.addWidget(self.user_edit, 1)
        conn_row.addWidget(self.connect_btn)
        conn_row.addWidget(self.disconnect_btn)

        # Messages view
        self.view = QListView()
        self.model = MessageListModel()
        self.view.setModel(self.model)
        self.view.setStyleSheet("font-size: 18px;")

        # Input row
        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Type a message and press Enter…")
        self.send_btn = QPushButton("Send")
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.send_btn)

        # Assemble
        main_layout.addLayout(conn_row)
        main_layout.addWidget(self.view, 1)
        main_layout.addLayout(input_row)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Wire UI -> emit intents
        self.connect_btn.clicked.connect(self._emit_connect)
        self.disconnect_btn.clicked.connect(self._emit_disconnect)
        self.send_btn.clicked.connect(self._emit_send)
        self.input_edit.returnPressed.connect(self._emit_send)

    # ---- Intents ----
    def _emit_disconnect(self):
        self.disconnectRequested.emit()
        
    def _emit_connect(self) -> None:
        host = self.host_edit.text().strip()
        user = self.user_edit.text().strip() or "Guest"
        try:
            port = int(self.port_edit.text().strip())
        except ValueError:
            self.alert("Port must be an integer.")
            return
        self.connectRequested.emit(host, port, user)

    def _emit_send(self) -> None:
        text = self.input_edit.text()
        self.input_edit.clear()
        self.sendRequested.emit(text)

    # ---- Minimal update API (controller/transport call these) ----
    def set_connected_ui(self, connected: bool) -> None:
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.host_edit.setEnabled(not connected)
        self.port_edit.setEnabled(not connected)
        self.user_edit.setEnabled(not connected)
        self.status.showMessage("Connected" if connected else "Disconnected", 3000)

    def add_local_echo(self, user: str, msg: str) -> None:
        t = datetime.now().strftime("%H:%M:%S")
        self.model.add_message(f"[{t}] {user}: {msg}")

    def add_remote_line(self, text: str) -> None:
        self.model.add_message(text)

    def alert(self, msg: str) -> None:
        QMessageBox.warning(self, "Chat", msg)


# ============================
# Controller (wires everything)
# ============================
class ChatController(QObject):
    def __init__(self, view: ChatWindow, connectionManager: ChatConnectionManager):
        super().__init__()
        self.view = view
        self.connectionManager = connectionManager

        # View intents -> controller actions
        self.view.connectRequested.connect(self._on_connect_requested)
        self.view.disconnectRequested.connect(self._on_disconnect_requested)
        self.view.sendRequested.connect(self._on_send_requested)

        # Transport events -> view updates
        self.connectionManager.connectedChanged.connect(self._on_connected_changed)
        self.connectionManager.errorOccurred.connect(self._on_error)
        self.connectionManager.lineReceived.connect(self.view.add_remote_line)

        # Initialize UI state
        self.view.set_connected_ui(False)

    # ---- Slots for view intents ----
    def _on_connect_requested(self, host: str, port: int, user: str) -> None:
        self.connectionManager.connect_to(host, port, user)

    def _on_disconnect_requested(self) -> None:
        self.connectionManager.disconnect()

    def _on_send_requested(self, text: str) -> None:
        if not text.strip():
            return
        self.connectionManager.send_message(text)
        # Optimistic echo; if your server also echos to sender, you can remove this line
        self.view.add_local_echo(self.connectionManager.username, text)

    # ---- Slots for connectionManager events ----
    def _on_connected_changed(self, connected: bool) -> None:
        self.view.set_connected_ui(connected)

    def _on_error(self, msg: str) -> None:
        self.view.alert(msg)


# ============================
# App entry
# ============================
def main() -> int:
    app = QApplication(sys.argv)
    view = ChatWindow()
    connectionManager = ChatConnectionManager()
    controller = ChatController(view, connectionManager)
    view.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
