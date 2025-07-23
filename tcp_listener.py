#!/usr/bin/env python3
import socket
import time

HOST, PORT = "127.0.0.1", 6000

def main():
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                data = s.recv(1024)     # получаем до 1024 байт
                if data:
                    pose = data.decode().strip()
                    print("Received pose:", pose)
        except ConnectionRefusedError:
            # сервер ещё не поднялся, пробуем снова
            pass
        time.sleep(0.1)  # пауза, чтобы не крутить цикл слишком плотно

if __name__ == "__main__":
    main()