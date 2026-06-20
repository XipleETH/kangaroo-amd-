# Pool Bridge — Collision Protocol

Bridge between the AMD Kangaroo solver and the [Collision Protocol](https://collisionprotocol.com) pool.

## Requirements

- Python 3.8+ (uses only stdlib — no pip install needed)
- Active internet connection

## Usage

```bash
python pool_bridge.py --worker YOUR_BTC_ADDRESS --dp-file dp_output.bin
```

## How it works

1. Connects to `pool.collisionprotocol.com:17403` via TLS
2. Authenticates with your Bitcoin address (this is where rewards go)
3. Receives work assignment from the pool
4. Reads distinguished points (DPs) from the binary file written by the solver
5. Submits DPs to the pool using JLP binary protocol v3
6. Handles reconnection, keepalive (PING/PONG), and error recovery

## DP File Format

Each DP record is 66 bytes:
```
[x_coordinate: 32 bytes BE] [distance: 32 bytes BE] [type: 1 byte] [dp_bits: 1 byte]
```

The solver appends DPs atomically. The bridge reads and submits them, tracking its position in the file.
