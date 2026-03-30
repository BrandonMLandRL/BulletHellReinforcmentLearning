# Unified-queue TCP learner server: shared _recv_queue / _send_queue mirroring Actor; broadcast send path.
import queue
import socket
import threading
from typing import Any, Callable, Dict, List, Optional

from bullet_hell_rl.net import protocol

HOST = "127.0.0.1"
PORT = 5556

OnMessage = Optional[Callable[[Dict[str, Any]], None]]


class LearnerServerComponent:
    """Accepts many actors; merges inbound JSON messages into _recv_queue; drains _send_queue to all clients."""

    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        on_message_callback: OnMessage = None,
    ):
        self._host = host
        self._port = port
        self._on_message = on_message_callback

        self._recv_queue: queue.Queue = queue.Queue()
        self._send_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        self._server_socket: Optional[socket.socket] = None
        self._client_sockets: List[socket.socket] = []
        self._client_lock = threading.Lock()

        self._accept_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None

    def _remove_client(self, conn: socket.socket) -> None:
        with self._client_lock:
            try:
                self._client_sockets.remove(conn)
            except ValueError:
                pass

    def _add_client(self, conn: socket.socket) -> None:
        with self._client_lock:
            self._client_sockets.append(conn)

    def _client_reader(self, conn: socket.socket, addr: Any) -> None:
        try:
            while not self._stop_event.is_set():
                msg = protocol.recv_message(conn)
                if msg is None:
                    break
                self._recv_queue.put(msg)
                if self._on_message is not None:
                    try:
                        self._on_message(msg)
                    except Exception as e:
                        print(f"Learner on_message callback error: {e}")
        finally:
            self._remove_client(conn)
            try:
                conn.close()
            except OSError:
                pass
            print(f"Actor disconnected from {addr}")

    def _accept_loop(self) -> None:
        assert self._server_socket is not None
        while not self._stop_event.is_set():
            try:
                conn, addr = self._server_socket.accept()
            except OSError:
                break
            print(f"Actor connected from {addr}")
            try:
                protocol.send_message(
                    conn,
                    {
                        "type": protocol.MSG_LEARNER_INIT,
                        "version": 0,
                        "message": "hello world",
                    },
                )
            except OSError:
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            self._add_client(conn)
            threading.Thread(
                target=self._client_reader,
                args=(conn, addr),
                name=f"LearnerRecv-{addr}",
                daemon=True,
            ).start()

    def _send_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._send_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                break
            with self._client_lock:
                targets = list(self._client_sockets)
            dead: List[socket.socket] = []
            for sock in targets:
                try:
                    protocol.send_message(sock, msg)
                except OSError:
                    dead.append(sock)
            for sock in dead:
                self._remove_client(sock)
                try:
                    sock.close()
                except OSError:
                    pass

    def start_server(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(16)
        print(f"Learner listening on {self._host}:{self._port}")

        self._stop_event.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop,
            name="LearnerSend",
            daemon=True,
        )
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="LearnerAccept",
            daemon=True,
        )
        self._send_thread.start()
        self._accept_thread.start()

        try:
            while self._accept_thread.is_alive():
                self._accept_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("Learner shutdown (KeyboardInterrupt)")
        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._send_queue.put(None)

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        with self._client_lock:
            clients = list(self._client_sockets)
            self._client_sockets.clear()
        for s in clients:
            try:
                s.close()
            except OSError:
                pass

        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)
        if self._accept_thread and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)
