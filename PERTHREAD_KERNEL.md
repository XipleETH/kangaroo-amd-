# Per-thread batch inversion kernel (AMD RDNA3 optimization)

This fork replaces the **workgroup-wide tree Montgomery inversion** in the Kangaroo
compute kernel with a **per-thread sequential Montgomery batch inversion**, the layout
used by the fastest CUDA solvers (RCKangaroo, JeanLucPons/Kangaroo, VanitySearch).

## Why

The original kernel inverts `dx = jump.x - px` with a workgroup-wide product tree:
~15 `workgroupBarrier()` per step and a **single `fe_inv` executed on lane 0 while the
other 127 lanes sit idle**. On AMD RDNA3 this shows up as *"GPU engine 99% busy, low
power draw, fans idle"* — a latency/occupancy-bound kernel, not a compute-bound one.

The new kernel gives **each thread its own group of `GROUP_N` kangaroos** and runs a
purely sequential Montgomery batch inversion (forward prefix products → one `fe_inv` →
backward pass). **Zero barriers, zero idle lanes** — every lane runs a full `fe_inv`
amortized over `GROUP_N` points, all in parallel.

## Measured result — AMD Radeon RX 7800 XT (RDNA3, 60 CU), Vulkan

| Kernel | Throughput | Per-step | Startup shader compile |
|---|---|---|---|
| Original tree (workgroup inversion) | ~30 M ops/s | 2.2 ms | ~15 min (first run) |
| **Per-thread N=4 (this fork)** | **~41 M ops/s** | **1.58 ms** | **~2.75 min (first run)** |

**~1.37× faster**, and as a side effect the AMD shader compile dropped from ~15 min to
~2.75 min (the barrier tree was pathologically slow to compile). After the first run the
AMD driver caches the compiled pipeline, so subsequent launches start instantly.

Correctness verified end-to-end against a known 56-bit target
(`pubkey 03a2efa4…de0d4`, `start 0x8000000000`, range 40 and 56): recovers the expected
key `0xe9ae4933d6` with `Verification: SUCCESS`.

## What changed

- `src/shaders/kangaroo_affine.wgsl` — full rewrite of `main()`. Removed the shared-memory
  product tree and all barriers; added `phase_a` (jump select + dx) and `phase_b_add`
  (negation-map add) helpers and a fully-unrolled `GROUP_N=4` group loop. All correctness
  invariants preserved verbatim: negation-map even-y representative, distance sign flip
  (`y_odd ? jump-dist : dist+jump`), per-point `dx==0` guard, jump index stashed **after**
  the anti-repeat/escape adjustment, one DP per kangaroo per dispatch.
- `src/solver.rs` — `const GROUP_N: u32 = 4` and dispatch geometry now launches
  `num_kangaroos / GROUP_N` threads (2 call sites). `ops_delta` is unchanged.
- `src/gpu/pipeline.rs` — `zero_initialize_workgroup_memory: false` (the new kernel has no
  `var<workgroup>` memory).

`GROUP_N` in the shader **must** match `GROUP_N` in `solver.rs`.

## Tuning `GROUP_N`

`GROUP_N` trades `fe_inv` amortization (larger N = fewer inversions) against VGPR pressure
(larger N = lower occupancy / register spill). On the 7800 XT, N=4 is the measured sweet
spot in registers; the peak register footprint during `fe_inv` (group state + Fermat
chain internals) is what caps it. To go materially higher you must move the group point
state out of registers into a coalesced global scratch buffer (the RCKangaroo approach) —
see "Future work".

## Future work (path to the card's real ceiling)

This WGSL kernel is register-bound at small N. The remaining headroom needs one of:

1. **Global/L2-backed group state** so `GROUP_N` can rise to 16–256 (far fewer `fe_inv`
   per point). This is the single biggest remaining lever inside WGSL.
2. **A HIP/ROCm port** with hand-tuned add-with-carry field arithmetic. WGSL lacks carry
   intrinsics, so 256-bit math is emulated with `u32` limbs; a HIP kernel would reach the
   ~1–2 Gkey/s that a 7800 XT is actually capable of (RCKangaroo does ~4 Gkey/s on a 3090).

## Build & run

```bash
cargo build --release
# example: puzzle #135 (public key known)
./target/release/kangaroo \
  --pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  --start 4000000000000000000000000000000000 --range 134
```

Set `RUST_LOG=info` to see the throughput (`Ops: …M`) and the `Verification: SUCCESS`
line. First launch spends ~2–3 min compiling the shader (AMD driver), then caches it.
