# kangaroo-hip — native HIP/ROCm port (AMD RDNA3) — WIP

A from-scratch native **HIP** (ROCm) port of the secp256k1 Pollard's-Kangaroo solver,
targeting AMD RDNA3 (gfx1101 / RX 7800 XT). The goal is to break past the ~50 M ops/s
ceiling of the WGSL/Vulkan implementation by using native `mul-hi` / carry arithmetic
that WGSL cannot express.

## Status

| Stage | State |
|---|---|
| 0. Feasibility (gfx1101 HIP support, Win10) | ✅ confirmed |
| 1. Toolchain + trivial kernel (`hello.hip`) | ✅ validated on RX 7800 XT |
| 2a. **256-bit field arithmetic** (`field_test.hip`) | ✅ **correct + fast** |
| 2b. **EC point ops** (`ec_test.hip`) | ✅ **correct, anchored to secp256k1** |
| 2c/3. **Full kangaroo solver** (`kangaroo.hip`) | ✅ **WORKS — recovers the 40-bit key** |
| 4. **Optimized batch-inversion solver** | ✅ **~1.17 G ops/s (~23× WGSL)** |

## Working solver (`kangaroo.hip`)

2-set (tame/wild) Pollard's Kangaroo, u64 distances. Recovers `k = 0xe9ae4933d6` from
`Q = k·G` on `[0, 2^40)` — first native-AMD kangaroo that solves an ECDLP.

**Speed progression on the RX 7800 XT (each step verified: recovers the key):**

| Version | ops/s | vs WGSL |
|---|---|---|
| WGSL/Vulkan (previous ceiling) | 50 M | 1× |
| HIP, per-step inversion | 12 M | 0.24× |
| + global-backed Montgomery batch inversion (GN=16) | 230 M | 4.6× |
| + Dettman `fe_inv` addition chain | 429 M | 8.6× |
| + GN=32 | 638 M | 12.8× |
| + GN=64 | 844 M | 17× |
| + more threads (65536) | **1174 M** | **~23×** |
| (peak, 131072 threads) | 1191 M | ~24× |

**~1.17 G ops/s** — in the same order of magnitude as an Nvidia RTX 3090 running RCKangaroo
(~4 G on a much larger card), i.e. competitive per-tier. What made it: global-backed group
state (registers spilled), the Dettman inversion chain, large GN, and heavy occupancy.
Native `fe_mul` is ~13.6 G/s, so remaining headroom is compute (a dedicated `fe_sqr` — the
255 squarings in `fe_inv` still call `fe_mul(a,a)`) and less global streaming.

TODO for real targets (#135): 256-bit distances (currently u64, ok to ~2^48), negation map,
pubkey-y recovery (modular sqrt), a `dx==0` guard, and cycle detection.

## Key result (Stage 2a) — the top risk is refuted

`field_test.hip` implements secp256k1 field arithmetic with `u64 x4` limbs and
`__int128` products (p = 2^256 − 2^32 − 977, folded via 2^256 ≡ 2^32+977). Measured on
an **RX 7800 XT**:

- **Correctness:** 1,638,400 algebraic self-tests — `(a+b)(a−b) == a²−b²` and
  `a·a⁻¹ == 1` — **0 failures.**
- **Throughput:** **~13,600 M `fe_mul`/s** (≈ **50×** the WGSL fe_mul rate).

clang lowers `__int128` to native `mul-hi` on RDNA3, so the "carry emulation is too
expensive" risk did **not** materialize. Rough kangaroo projection ≈ mul_rate/5 ≈
2.7 G ops/s; the real kernel (memory/branches/inversion) will land lower, realistically
**~1–2 G ops/s = 20–40× over WGSL** — to be confirmed once the full kernel is built.
This is, as far as we know, the first AMD-native kangaroo field-arithmetic data point.

## Toolchain

- AMD HIP SDK for Windows **7.1** (`C:\Program Files\AMD\ROCm\7.1`) + Visual Studio 2022
  (MSVC host compiler). gfx1101 is officially supported on Windows 10/11.
- The HIP SDK's bundled display-driver component may throw Error 1603 (its ReLive/Settings
  sub-packages) — that is **irrelevant to compute**; the pre-existing Adrenalin driver
  already ships the HIP runtime. Leave the driver at "Don't Install".

## Build & run

Open the **HIP SDK Command Prompt** (or a shell with `%HIP_PATH%\bin` on PATH), then:

```bat
build_field.bat        REM compiles field_test.hip; then run field_test.exe separately
```

`build_*.bat` loads `vcvars64.bat` (adjust the VS path if not Community) and calls
`hipcc -O3 --offload-arch=gfx1101`. See `SETUP.md` for the install steps.
