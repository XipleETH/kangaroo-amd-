# 🦘 Kangaroo-AMD

**GPU-accelerated Pollard's Kangaroo solver for the Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1 — with AMD GPU support.**

> Fork of [oritwoen/kangaroo](https://github.com/oritwoen/kangaroo) by [XipleETH](https://github.com/XipleETH/kangaroo-amd-)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

The original kangaroo project only supported NVIDIA GPUs. This fork adds **AMD GPU support** (tested on RDNA3 / RX 7800 XT) using [wgpu](https://wgpu.rs/) (WebGPU) for cross-platform GPU compute via Vulkan and DX12.

### Key features

- **Two-pass GPU pipeline** optimized for AMD shader compilers
  - *Walk shader* — Jacobian coordinates, no `fe_inv` (avoids RDNA3 compiler hangs)
  - *Normalize shader* — field inversion + exact DP detection in a separate, simpler pass
- **~560M ops/s** on AMD Radeon RX 7800 XT
- **Pool bridge** for [Collision Protocol](https://collisionprotocol.com) mining
- **CPU fallback** for systems without a compatible GPU
- Double-buffered GPU pipeline for maximum throughput

---

## Quick Start

### 1. Local Solo Mode (default)

Run the solver independently to find a private key.

```
kangaroo.exe --pubkey <COMPRESSED_PUBKEY> --range <BITS> --dp-bits <DP_BITS>
```

**Example — Bitcoin Puzzle #40:**

```
kangaroo.exe --pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 --range 40 --dp-bits 10 --kangaroos 65536
```

### 2. Pool Mode (Collision Protocol)

Connect to the [Collision Protocol](https://collisionprotocol.com) pool to contribute work alongside other miners. Requires two processes: the solver writing DPs to a file, and the Python bridge submitting them to the pool.

**Step 1 — Start the bridge:**

```
python bridge/pool_bridge.py --worker YOUR_BTC_ADDRESS --dp-file dp_output.bin
```

**Step 2 — Start the solver:**

```
kangaroo.exe --pubkey <POOL_PUBKEY> --range 135 --kangaroos 65536 --dp-bits 28 --mode wild --dp-output dp_output.bin
```

Or use the batch script: **`run_pool.bat`** (edit your wallet address first).

### 3. CPU Mode (no GPU needed)

```
kangaroo.exe --pubkey <PUBKEY> --range <BITS> --cpu
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `-p, --pubkey` | *required* | Target public key (compressed hex, 33 bytes) |
| `-r, --range` | *required* | Bit range to search |
| `-s, --start` | auto | Start of search range (hex) |
| `-d, --dp-bits` | auto | Distinguished point bits (lower = more DPs, more memory) |
| `-k, --kangaroos` | auto | Number of parallel kangaroos |
| `--mode` | `both` | Kangaroo mode: `both` (solo), `tame`, or `wild` (pool) |
| `--dp-output` | — | Write DPs to binary file (for pool bridge) |
| `--gpu` | `0` | GPU index, comma-separated, or `all` |
| `--list-gpus` | | List available GPU devices |
| `--backend` | `auto` | GPU backend: `auto` / `vulkan` / `dx12` / `metal` / `gl` |
| `--cpu` | `false` | Use CPU solver instead of GPU |
| `-o, --output` | — | Output file for found key |
| `-q, --quiet` | `false` | Minimal output |
| `--max-ops` | unlimited | Maximum operations before stopping |
| `-t, --target` | — | Data provider (e.g. `boha:b1000/135`) |
| `--list-providers` | | List available puzzle providers |
| `--benchmark` | `false` | Run benchmark suite |

---

## Architecture

The solver uses a **two-pass GPU compute pipeline** designed to work around AMD RDNA3 shader compiler limitations:

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU Slot N                          GPU Slot N-1               │
│  ┌──────────────────┐                ┌──────────────────┐       │
│  │  1. Walk Shader  │  dispatch      │  Read back DPs   │       │
│  │  (Jacobian, no   │◄──────────►    │  from previous   │       │
│  │   fe_inv)        │                │  dispatch         │       │
│  └────────┬─────────┘                └──────────────────┘       │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  2. Normalize    │                                           │
│  │  Shader (fe_inv  │                                           │
│  │  + DP detection) │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

1. **Walk Shader** (`kangaroo_jacobian.wgsl`) — Performs N steps of the kangaroo walk in Jacobian coordinates. No field inversion needed, which avoids the complex `fe_inv` function that causes AMD RDNA3 shader compilers to hang.

2. **Normalize Shader** (`normalize_dp.wgsl`) — Normalizes all kangaroos from Jacobian (X, Y, Z) to Affine (X/Z², Y/Z³) coordinates using `fe_inv`. Checks the exact DP condition on the affine X coordinate. This shader is simple enough to compile on AMD without issues.

The pipeline is **double-buffered**: it dispatches walk + normalize on slot N while reading back results from slot N−1.

---

## Performance

Tested on **AMD Radeon RX 7800 XT** (RDNA3):

| Metric | Value |
|---|---|
| Throughput | **~560M ops/s** |
| Kangaroos | 65,536 |
| Steps per dispatch | 512 |
| Workgroup size | 64 |

### Bitcoin Puzzle Solving Times

All results verified correct:

| Puzzle | Bits | Time |
|---|---|---|
| #20 – #32 | 20–32 | < 3 s |
| #33 – #37 | 33–37 | 5–15 s |
| #38 – #42 | 38–42 | 38 s – 3 min |
| #43 – #47 | 43–47 | 5–23 min |

---

## Building from Source

### Prerequisites

- **Rust 1.75+** — `rustup install stable`
- **Vulkan SDK** or AMD GPU drivers with Vulkan support
- **Python 3.8+** (for pool bridge only)

### Build

```bash
git clone https://github.com/XipleETH/kangaroo-amd-.git
cd kangaroo-amd-
cargo build --release
```

The binary will be at:
- **Windows:** `target/release/kangaroo.exe`
- **Linux:** `target/release/kangaroo`

> [!NOTE]
> The GPU normalization shader takes **~46 seconds** to compile the first time on AMD. Subsequent runs use the driver's shader cache and start instantly.

---

## Pool Bridge

The pool bridge (`bridge/pool_bridge.py`) connects to the Collision Protocol pool at `pool.collisionprotocol.com:17403` using TLS.

- **Protocol:** JLP binary wire protocol v3
- **No dependencies:** Uses only Python stdlib (`ssl`, `socket`, `struct`)

### How it works

1. Bridge authenticates with your Bitcoin address
2. Pool assigns work (pubkey, range, dp_bits, kangaroo type)
3. Solver writes DPs to a binary file as it finds them
4. Bridge reads DPs and submits them to the pool
5. If a collision is found, the pool distributes the reward

---

## Donations

If you find this fork useful, donations are welcome:

**BTC:** `bc1qxtyjnyszrsvcwndvzsx6ee7s7hm5sg8uzl8duq`

---

## Credits

- **Original project:** [oritwoen/kangaroo](https://github.com/oritwoen/kangaroo)
- **AMD GPU support, GPU normalization shader, and pool bridge:** [XipleETH](https://github.com/XipleETH)
- Based on Pollard's Kangaroo algorithm for ECDLP
- Pool: [Collision Protocol](https://collisionprotocol.com)

## License

MIT License — see [LICENSE](LICENSE) for details.
