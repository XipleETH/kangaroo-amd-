#!/usr/bin/env python3
"""
AMD Kangaroo -> Collision Protocol Pool Bridge

Connects our AMD GPU kangaroo solver to the Collision Protocol pool
for Bitcoin Puzzle #135 using the JLP binary wire protocol over TLS.

Protocol reference: https://github.com/hevnsnt/collider/blob/main/docs/JLP-PROTOCOL.md
"""

import socket
import ssl
import struct
import time
import os
import sys
import threading
import argparse
import secrets
from pathlib import Path

# ============================================================================
# JLP Protocol Constants (from JLP-PROTOCOL.md)
# ============================================================================

POOL_HOST = "pool.collisionprotocol.com"
POOL_PORT = 17403

# Frame header magic
MAGIC = b'KANG'

# Message types
MSG_AUTH         = 0x01
MSG_AUTH_OK      = 0x02
MSG_AUTH_FAIL    = 0x03
MSG_WORK_REQ     = 0x10
MSG_WORK_ASN     = 0x11
MSG_DP_SUBMIT    = 0x20
MSG_DP_ACK       = 0x21
MSG_DP_BATCH     = 0x22
MSG_DP_SUBMIT_V2 = 0x23
MSG_DP_BATCH_V2  = 0x24
MSG_DP_SUBMIT_V3 = 0x25
MSG_DP_BATCH_V3  = 0x26
MSG_STATS_REQ    = 0x30
MSG_STATS_RSP    = 0x31
MSG_CHALLENGE    = 0x32
MSG_CHALLENGE_RSP = 0x33
MSG_SOLUTION     = 0x40
MSG_PING         = 0x50
MSG_PONG         = 0x51
MSG_MAINTENANCE  = 0x60
MSG_ERROR        = 0xFF

# Protocol version (we present as v3 to avoid v4 challenges)
PROTO_VERSION = 3

# Kangaroo types (from protocol)
KTYPE_BOTH = 0       # reserved, illegal in pool mode
KTYPE_TAME_ONLY = 1
KTYPE_WILD_ONLY = 2

# Sizes
AUTH_PAYLOAD_SIZE = 120   # AuthPayloadV2
WORK_ASN_SIZE = 126       # WorkAssignment
DP_V2_SIZE = 78           # DistinguishedPointV2
MAX_DPS_PER_BATCH = 10000

MSG_NAMES = {
    MSG_AUTH: "AUTH", MSG_AUTH_OK: "AUTH_OK", MSG_AUTH_FAIL: "AUTH_FAIL",
    MSG_WORK_REQ: "WORK_REQ", MSG_WORK_ASN: "WORK_ASN",
    MSG_DP_ACK: "DP_ACK", MSG_DP_BATCH_V2: "DP_BATCH_V2",
    MSG_STATS_RSP: "STATS_RSP", MSG_CHALLENGE: "CHALLENGE",
    MSG_SOLUTION: "SOLUTION", MSG_PING: "PING", MSG_PONG: "PONG",
    MSG_MAINTENANCE: "MAINTENANCE", MSG_ERROR: "ERROR",
}

# ============================================================================
# JLP Frame Header (8 bytes)
# Layout: KANG(4) + type(1) + flags(1) + payload_size(2 LE)
# ============================================================================

HEADER_SIZE = 8
HEADER_FMT = '<4sBBH'  # magic(4) + type(u8) + flags(u8) + payload_size(u16)

def pack_header(msg_type: int, payload_size: int, version: int = PROTO_VERSION) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, msg_type, version, payload_size)

def unpack_header(data: bytes) -> tuple:
    """Returns (msg_type, flags/version, payload_size)."""
    magic, msg_type, flags, payload_size = struct.unpack(HEADER_FMT, data)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")
    return msg_type, flags, payload_size

# ============================================================================
# JLP Messages
# ============================================================================

def pack_auth(worker_name: str, password: str = "") -> bytes:
    """Pack AUTH (0x01) message with AuthPayloadV2 (120 bytes)."""
    # <64s32sQ16s
    name_bytes = worker_name.encode('utf-8')[:64].ljust(64, b'\x00')
    pass_bytes = password.encode('utf-8')[:32].ljust(32, b'\x00')
    timestamp_ms = int(time.time() * 1000)
    nonce = secrets.token_bytes(16)
    
    payload = struct.pack('<64s32sQ16s', name_bytes, pass_bytes, timestamp_ms, nonce)
    assert len(payload) == AUTH_PAYLOAD_SIZE, f"Auth size {len(payload)} != {AUTH_PAYLOAD_SIZE}"
    
    return pack_header(MSG_AUTH, AUTH_PAYLOAD_SIZE) + payload

def pack_work_req() -> bytes:
    """Pack WORK_REQ (0x10) message - request work assignment."""
    return pack_header(MSG_WORK_REQ, 0)

def pack_ping() -> bytes:
    """Pack PING (0x50) keepalive."""
    return pack_header(MSG_PING, 0)


class WorkAssignment:
    """Parsed WorkAssignment (0x11), 126 bytes payload."""

    def __init__(self, data: bytes):
        offset = 0
        self.pubkey = data[offset:offset+33]; offset += 33
        self.range_start = data[offset:offset+32]; offset += 32
        self.range_end = data[offset:offset+32]; offset += 32
        self.dp_bits = struct.unpack_from('<I', data, offset)[0]; offset += 4
        self.work_id = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        self.kangaroo_type = data[offset]; offset += 1
        # start_offset_a (u64) and start_offset_b (u64) if present
        if len(data) >= offset + 8:
            self.start_offset_a = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        else:
            self.start_offset_a = 0
        if len(data) >= offset + 8:
            self.start_offset_b = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        else:
            self.start_offset_b = 0

    @property
    def type_name(self):
        return {KTYPE_TAME_ONLY: "TAME", KTYPE_WILD_ONLY: "WILD"}.get(self.kangaroo_type, f"UNKNOWN({self.kangaroo_type})")

    def __str__(self):
        start_hex = self.range_start.hex().lstrip('0') or '0'
        end_hex = self.range_end.hex().lstrip('0') or '0'
        return (
            f"  Pubkey:     {self.pubkey.hex()}\n"
            f"  Range:      0x{start_hex} .. 0x{end_hex}\n"
            f"  DP bits:    {self.dp_bits}\n"
            f"  Work ID:    {self.work_id}\n"
            f"  Type:       {self.type_name}\n"
            f"  Offset A:   {self.start_offset_a}\n"
            f"  Offset B:   {self.start_offset_b}"
        )


def pack_dp_batch_v2(dps: list) -> bytes:
    """Pack DP_BATCH_V2 (0x24) message.
    
    Each DP dict needs: work_id, sequence, x(32B), d(32B), type(u8), dp_bits(u8)
    DP wire size: 8 + 4 + 32 + 32 + 1 + 1 = 78 bytes
    """
    payload = b''
    for dp in dps:
        dp_bytes = struct.pack('<QI', dp['work_id'], dp['sequence'])
        dp_bytes += dp['x'][:32].rjust(32, b'\x00')  # 32 bytes BE
        dp_bytes += dp['d'][:32].rjust(32, b'\x00')   # 32 bytes BE
        dp_bytes += struct.pack('BB', dp['type'], dp['dp_bits'])
        payload += dp_bytes

    return pack_header(MSG_DP_BATCH_V2, len(payload)) + payload


# ============================================================================
# Pool Connection
# ============================================================================

class PoolConnection:
    """Manages TLS connection to Collision Protocol pool."""

    def __init__(self, host: str, port: int, worker_name: str):
        self.host = host
        self.port = port
        self.worker_name = worker_name
        self.sock = None
        self.work = None
        self.sequence = 0
        self.dps_submitted = 0
        self.connected = False
        self.negotiated_version = PROTO_VERSION

    def connect(self):
        """Establish TLS connection and authenticate."""
        print(f"[POOL] Connecting to {self.host}:{self.port} (TLS)...")
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(60)
        self.sock = ctx.wrap_socket(raw_sock, server_hostname=self.host)
        self.sock.connect((self.host, self.port))
        print(f"[POOL] TLS connected!")

        # Send AUTH
        print(f"[POOL] Sending AUTH as '{self.worker_name}' (proto v{PROTO_VERSION})...")
        auth_msg = pack_auth(self.worker_name)
        self.sock.sendall(auth_msg)

        # Wait for AUTH_OK or AUTH_FAIL
        msg_type, flags, payload = self._recv_frame()
        self.negotiated_version = flags
        
        if msg_type == MSG_AUTH_OK:
            print(f"[POOL] AUTH_OK! (negotiated v{flags}, payload={len(payload)}b)")
        elif msg_type == MSG_AUTH_FAIL:
            reason = payload.decode('utf-8', errors='replace') if payload else "unknown"
            raise ConnectionError(f"AUTH_FAIL: {reason}")
        else:
            print(f"[POOL] Unexpected response: {MSG_NAMES.get(msg_type, hex(msg_type))} ({len(payload)}b)")

        # Request work
        print(f"[POOL] Requesting work assignment...")
        self.sock.sendall(pack_work_req())
        
        # Wait for WORK_ASN
        self.work = self._wait_for_work()
        if self.work:
            print(f"[POOL] Work received!\n{self.work}")
            self.connected = True
            self.sequence = 0
        else:
            raise ConnectionError("No work assignment received")

    def _recv_exact(self, n: int) -> bytes:
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def _recv_frame(self) -> tuple:
        """Receive one JLP frame. Returns (msg_type, flags, payload)."""
        header = self._recv_exact(HEADER_SIZE)
        msg_type, flags, payload_size = unpack_header(header)
        payload = self._recv_exact(payload_size) if payload_size > 0 else b''
        return msg_type, flags, payload

    def _wait_for_work(self) -> WorkAssignment:
        """Wait for WORK_ASN, handling other messages."""
        for _ in range(20):  # max 20 messages before giving up
            msg_type, flags, payload = self._recv_frame()
            name = MSG_NAMES.get(msg_type, f"0x{msg_type:02x}")
            
            if msg_type == MSG_WORK_ASN:
                return WorkAssignment(payload)
            elif msg_type == MSG_PONG:
                print(f"[POOL] PONG received")
            elif msg_type == MSG_STATS_RSP:
                print(f"[POOL] Stats received ({len(payload)}b)")
            elif msg_type == MSG_ERROR:
                err = payload.decode('utf-8', errors='replace')
                print(f"[POOL] ERROR: {err}")
            else:
                print(f"[POOL] Received {name} ({len(payload)}b), waiting for WORK_ASN...")
        return None

    def submit_dps(self, dps: list) -> int:
        """Submit a batch of DPs. Returns number submitted."""
        if not self.connected or not self.work:
            return 0

        for dp in dps:
            dp['work_id'] = self.work.work_id
            dp['sequence'] = self.sequence
            dp['type'] = self.work.kangaroo_type
            dp['dp_bits'] = self.work.dp_bits
            self.sequence += 1

        msg = pack_dp_batch_v2(dps)
        self.sock.sendall(msg)
        self.dps_submitted += len(dps)
        return len(dps)

    def check_messages(self):
        """Non-blocking check for incoming messages."""
        self.sock.settimeout(0.1)
        try:
            msg_type, flags, payload = self._recv_frame()
            name = MSG_NAMES.get(msg_type, f"0x{msg_type:02x}")
            
            if msg_type == MSG_WORK_ASN:
                self.work = WorkAssignment(payload)
                self.sequence = 0
                print(f"[POOL] New work assignment!\n{self.work}")
            elif msg_type == MSG_DP_ACK:
                pass  # silently acknowledge
            elif msg_type == MSG_PING:
                self.sock.sendall(pack_header(MSG_PONG, 0))
            elif msg_type == MSG_SOLUTION:
                print(f"\n{'='*60}")
                print(f"  !!!  SOLUTION FOUND  !!!")
                print(f"  The puzzle has been solved!")
                print(f"{'='*60}\n")
            elif msg_type == MSG_ERROR:
                err = payload.decode('utf-8', errors='replace')
                print(f"[POOL] ERROR: {err}")
            else:
                print(f"[POOL] {name} ({len(payload)}b)")
        except socket.timeout:
            pass
        except ssl.SSLWantReadError:
            pass
        finally:
            self.sock.settimeout(60)

    def reconnect(self, wait=10):
        """Reconnect on failure with backoff."""
        self.connected = False
        try:
            self.sock.close()
        except:
            pass
        print(f"[POOL] Reconnecting in {wait}s...", flush=True)
        time.sleep(wait)
        for attempt in range(5):
            try:
                self.connect()
                return
            except Exception as e:
                wait_time = min(30, 10 * (attempt + 1))
                print(f"[POOL] Retry {attempt+1}/5 failed: {e}. Waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
        print("[POOL] All retries failed, will try again next cycle", flush=True)


# ============================================================================
# DP File Reader  
# ============================================================================

class DPFileReader:
    """Reads DPs from binary file written by kangaroo solver.
    
    Format per DP (66 bytes):
      [x: 32B BE] [d: 32B BE] [type: 1B] [dp_bits: 1B]
    """
    RECORD_SIZE = 66

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.offset = 0

    def read_new_dps(self) -> list:
        dps = []
        try:
            fsize = os.path.getsize(self.filepath)
            if fsize <= self.offset:
                return dps
            with open(self.filepath, 'rb') as f:
                f.seek(self.offset)
                while True:
                    rec = f.read(self.RECORD_SIZE)
                    if len(rec) < self.RECORD_SIZE:
                        break
                    dps.append({
                        'x': rec[0:32],
                        'd': rec[32:64],
                        'type': rec[64],
                        'dp_bits': rec[65],
                        'work_id': 0,
                        'sequence': 0,
                    })
                    self.offset += self.RECORD_SIZE
        except FileNotFoundError:
            pass
        return dps


# ============================================================================
# Main Bridge
# ============================================================================

def run_bridge(worker: str, dp_file: str, host: str, port: int):
    print("=" * 60)
    print("  AMD Kangaroo -> Collision Protocol Bridge")
    print("=" * 60)
    print(f"  Worker:   {worker}")
    print(f"  Pool:     {host}:{port}")
    print(f"  DP file:  {dp_file}")
    print(f"  Protocol: JLP v{PROTO_VERSION}")
    print("=" * 60)

    pool = PoolConnection(host, port, worker)
    
    # Initial connection with retry
    for attempt in range(10):
        try:
            pool.connect()
            break
        except Exception as e:
            wait_time = min(60, 15 * (attempt + 1))
            print(f"[POOL] Connect attempt {attempt+1} failed: {e}", flush=True)
            print(f"[POOL] Retrying in {wait_time}s...", flush=True)
            time.sleep(wait_time)

    ktype = pool.work.type_name
    print(f"\n[BRIDGE] Assigned: {ktype} kangaroos")
    print(f"[BRIDGE] DP bits: {pool.work.dp_bits}")
    print(f"[BRIDGE] Work ID: {pool.work.work_id}")
    print(f"\n[BRIDGE] Start solver with: --mode {ktype.lower()} --dp-output {dp_file}")
    print()

    reader = DPFileReader(dp_file)
    t0 = time.time()
    last_report = t0
    last_ping = t0

    while True:
        try:
            # Read new DPs from solver
            dps = reader.read_new_dps()
            if dps:
                for i in range(0, len(dps), MAX_DPS_PER_BATCH):
                    batch = dps[i:i+MAX_DPS_PER_BATCH]
                    n = pool.submit_dps(batch)
                    if n > 0:
                        print(f"[BRIDGE] Submitted {n} DPs (total: {pool.dps_submitted:,})", flush=True)

            # Check for pool messages
            pool.check_messages()

            now = time.time()

            # Send PING keepalive every 15s
            if now - last_ping >= 15:
                try:
                    pool.sock.sendall(pack_ping())
                    last_ping = now
                except:
                    pass

            # Stats every 30s
            if now - last_report >= 30:
                elapsed = now - t0
                rate = pool.dps_submitted / elapsed if elapsed > 0 else 0
                print(f"[STATS] DPs: {pool.dps_submitted:,} | Rate: {rate:.2f}/s | Up: {elapsed/60:.1f}m", flush=True)
                last_report = now

            time.sleep(0.5)

        except (ConnectionError, BrokenPipeError, ssl.SSLError, OSError) as e:
            print(f"[POOL] Connection error: {e}", flush=True)
            pool.reconnect()
        except KeyboardInterrupt:
            print(f"\n[BRIDGE] Shutdown. Total DPs: {pool.dps_submitted:,}", flush=True)
            break


def main():
    p = argparse.ArgumentParser(description="AMD Kangaroo -> Collision Protocol Bridge")
    p.add_argument("--worker", "-w", required=True, help="Bitcoin payout address")
    p.add_argument("--dp-file", "-f", default="dp_output.bin", help="DP binary file path")
    p.add_argument("--host", default=POOL_HOST)
    p.add_argument("--port", "-p", type=int, default=POOL_PORT)
    args = p.parse_args()
    run_bridge(args.worker, args.dp_file, args.host, args.port)


if __name__ == "__main__":
    main()
