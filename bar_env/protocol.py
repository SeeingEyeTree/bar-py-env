"""Wire protocol between the Python env and the Lua bridge widget.

Each message is a 4-byte big-endian length prefix followed by UTF-8 JSON.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MB safety cap, must match widget


class ProtocolError(Exception):
    pass


def send_msg(sock: socket.socket, msg: dict[str, Any]) -> None:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message too large: {len(body)} bytes")
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    parts: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ProtocolError("connection closed mid-message")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    hdr = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", hdr)
    if length == 0 or length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"implausible message length {length}")
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))
