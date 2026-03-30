#This file will handle creating a TCP server that Actors will connect to which the Actors will send experience tuples to.
#Learner will then train using DQN on the incoming data. Learner will broadcast to all Actors when it has finished updating the weights file. 
#Actors should use a counting semaphore to prevent the Learner from updating the weights again while an Actor is reading still. 

import socket
import threading
from bullet_hell_rl.net import protocol


HOST = "127.0.0.1"
PORT = 5556

def main(host: str = HOST, port: int = PORT):
    start_server(host=host, port=port)

def start_server(host: str = HOST, port: int = PORT):
    """Start a persistent TCP server that accepts multiple actors."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(16)
    print(f"Learner listening on {host}:{port}")
    clients = []

    def handle_client(conn, addr):
        print(f"Actor connected from {addr}")
        try:
            # Test handshake: actor has already consumed MSG_WELCOME in its __init__
            protocol.send_message(
                conn,
                {
                    "type": protocol.MSG_WEIGHTS_READY,
                    "version": 0,
                    "note": "test_weights_ready",
                },
            )
            while True:
                msg = protocol.recv_message(conn)
                if msg is None:
                    break
                if msg.get("type") == protocol.MSG_WEIGHTS_READY_ACK:
                    print(f"Learner: received MSG_WEIGHTS_READY_ACK from {addr}: {msg}")
                    break
        except OSError:
            pass
        finally:
            try:
                conn.close()
            finally:
                if conn in clients:
                    clients.remove(conn)
            print(f"Actor disconnected from {addr}")

    try:
        while True:
            conn, addr = server_socket.accept()
            clients.append(conn)
            protocol.send_message(
                conn,
                {
                    "type": protocol.MSG_WELCOME,
                    "message": "hello world",
                },
            )
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()