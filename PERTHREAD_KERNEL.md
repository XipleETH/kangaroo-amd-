# Per-thread batch inversion kernel (AMD RDNA3 optimization)

This fork replaces the **workgroup-wide tree Montgomery inversion** in the Kangaroo
compute kernel with a **per-thread sequential Montgomery batch inversion**, the layout
used by the fastest CUDA solvers (RCKangaroo, JeanLucPons/Kangaroo, VanitySearch).

## Why

The original kernel inverts `dx = jump.x - px` with a workgroup-wide product tree:
~15 `workgroupBarrier()` per step and a **single `fe_inv` executed on lane 0 while the
other 127 lanes sit idle**. On AMD RDNA3 this shows up as *"GPU engine 99% busy, low
power draw, fans idle"* — a latency/occupancy-bound kernel, not compute-bound.

Two designs were implemented (both preserve the walk exactly; see "Correctness"):

1. **Per-thread, registers (`GROUP_N=4`)** — each thread owns 4 kangaroos, group state
   in registers, one `fe_inv` per 4 points. Zero barriers. Register-bound at small N.
2. **Per-thread, global-backed (`GROUP_N=16`, shipped)** — group point state lives in
   the global `kangaroos` buffer (streamed per step); a small scratch buffer holds the
   prefix products; only the rolling inverse stays in registers. This keeps the register
   footprint low (high occupancy) **and** amortizes the single `fe_inv` over a large
   `GROUP_N` (few inversions). This is the RCKangaroo memory layout.

## Measured result — AMD Radeon RX 7800 XT (RDNA3, 60 CU), Vulkan

| Kernel | Throughput (sustained) | Peak (calibration) | vs original |
|---|---|---|---|
| Original tree (workgroup inversion) | ~30 M ops/s | ~30 M ops/s | 1.0× |
| Per-thread `GROUP_N=4` (registers) | ~41 M ops/s | ~41 M ops/s | ~1.4× |
| **Per-thread `GROUP_N=16` (global-backed, shipped)** | **~50 M ops/s** | **~59 M ops/s** | **~1.7–2.0×** |

`GROUP_N=32` was tested and regressed (~42 M ops/s) — the larger unrolled loop raises
register pressure again; **`GROUP_N=16` is the RDNA3 sweet spot**.

As a side effect the AMD first-run shader compile dropped from ~15 min (the barrier tree
was pathologically slow to compile) to ~1–3 min; the driver then caches the pipeline and
subsequent launches start instantly.

### End-to-end solve scaling (verified, GROUP_N=16)

Fixed test key `0xe9ae4933d6` searched over increasingly large intervals
(`--pubkey 03a2efa4…de0d4 --start 8000000000 --range R`), all recover the key with
`Verification: SUCCESS`:

| range R | wall time |
|---|---|
| 40 | 0.37 s |
| 44 | 2.12 s |
| 48 | 12.07 s |
| 52 | 68.27 s |
| 56 | 401.97 s |

Time grows ~5.8× per +4 range bits (√N scaling × the kangaroo constant). Note this is the
solver's *interval-search cost*, which is exponential in range and unchanged by this work;
the **throughput** above is the kernel-power metric this optimization improves.

## What changed

- `src/shaders/kangaroo_affine.wgsl` — rewrite of `main()`. Removed the shared-memory
  product tree and all barriers. Each thread walks `GROUP_N` kangaroos via a two-pass
  sequential Montgomery batch inversion: PASS 1 folds `dx` into a running product and
  stores prefix products to scratch; one `fe_inv`; PASS 2 re-runs the (deterministic,
  side-effect-free) jump selection, recovers `1/dx_i`, does the negation-map add, and
  writes the kangaroo back. `phase_a`/`phase_b_add` helpers isolate selection and add.
- `src/gpu/mod.rs` — `pub const GROUP_N: u32 = 16` (single source of truth; the shader
  const must match).
- `src/solver.rs` — dispatch launches `num_kangaroos / GROUP_N` threads.
- `src/gpu/buffers.rs` + `src/gpu/pipeline.rs` — new binding 6: a prefix-product scratch
  buffer (`num_threads * GROUP_N` field elements, SoA-indexed for coalescing), shared
  across the two DP slots (GPU dispatches run in submission order, never concurrent).
  `zero_initialize_workgroup_memory = false` (kernel has no `var<workgroup>` memory).

## Correctness

Preserved verbatim from the original kernel and verified end-to-end (recovers
`0xe9ae4933d6` at ranges 40–56):

- negation-map even-y class representative and the distance sign flip
  (`y_odd ? scalar_sub_256(jd,dist) : scalar_add_256(dist,jd)`);
- per-point `dx==0` guard (substitute 1 before folding, so a collision can't zero the
  running product);
- jump index taken **after** the anti-repeat/escape adjustment, and identical in both
  passes because selection is deterministic and the global state is unchanged between
  passes;
- one distinguished point per kangaroo per dispatch (per-thread `u32` bitmask);
- `GpuKangaroo` (128 B) and all struct layouts unchanged — `bytemuck` readback stays valid.

## Tuning `GROUP_N`

Trades `fe_inv` amortization (larger N = fewer inversions) against register/loop pressure.
Measured on the 7800 XT: N=4 → 41 M, N=16 → 50 M (best), N=32 → 42 M. Change it in **both**
`src/gpu/mod.rs` and the shader `const GROUP_N`, keep it ≤ 32 (DP bitmask is one `u32`).

## Future work (path to the card's real ceiling)

WGSL emulates 256-bit math with `u32` limbs (no carry intrinsics), so it can't match a
hand-tuned CUDA/HIP kernel. A **HIP/ROCm port** with add-with-carry field arithmetic would
reach the ~1–2 Gkey/s a 7800 XT is actually capable of (RCKangaroo does ~4 Gkey/s on a 3090).

## Build & run

```bash
cargo build --release
RUST_LOG=info ./target/release/kangaroo \
  --pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  --start 4000000000000000000000000000000000 --range 134   # puzzle #135
```

First launch spends ~1–3 min compiling the shader (AMD driver), then caches it.
