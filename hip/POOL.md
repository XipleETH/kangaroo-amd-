# Trustless(-minimized) mixed AMD/Nvidia kangaroo pool — design + P0

Goal: let **AMD** (this HIP kangaroo) and **Nvidia** (RCKangaroo) clients jointly attack an
exposed-pubkey puzzle (e.g. #135), pool their distinguished points (DPs), detect the
collision across the whole swarm, and split the prize **by contribution** with the **least
possible trust**.

## The honest custody verdict (read this first)

A Bitcoin private key is **atomic** — whoever assembles it controls *all* the coins. There is
no native way to give a contributor "1% of spend authority". So:

- **A genuinely trustless fair split is NOT achievable** for a single-key puzzle. The honest
  target is **trust-minimized**: the key is reconstructed inside a **t-of-n MPC / threshold
  signature** so it never materializes on one machine, and the quorum's only output is a
  signature on **one pre-agreed split transaction** (outputs = each contributor's share).
- **Two separate trust residues** remain: (1) the MPC quorum not colluding; (2) a DP proves
  endpoint-consistency (`dist·G == P` for tame, `Q + dist·G == P` for wild) but **not** that
  the client did the work — fake DPs cost the same to mint, so contribution accounting falls
  back to statistics + stake, never proof.
- **Operator ≠ sole custodian.** You can *run* the pool (server, coordination, reputation)
  while being just **one of the N keys** — so even you cannot take the prize alone. That is
  how you *generate* trust: not by promising, by architecture.
- **Front-running:** a correctly-signed sweep tx does **not** leak the private key, so a
  mempool watcher cannot steal it. Keep the quorum "hot" (signs in seconds), submit the final
  signed tx via a **private relay / miner** (not the public mempool), high fee, atomic. The
  residual risk is a miner who *independently* has `k` — negligible unless they solved it too.

## P0 — DONE ✅ : cross-client DP pooling proven on one AMD card

`pool_demo.hip` proves the pool's core mechanic **without needing an Nvidia card**, by running
two independent clients on the same GPU:

- **client 0** runs ONLY *tame* kangaroos → writes its DPs to `c0.dp`
- **client 1** runs ONLY *wild* kangaroos → writes its DPs to `c1.dp`
- **neither can solve alone**; the **detector** merges both files, finds a
  tame(client0)↔wild(client1) collision, recovers `k`, and reports which clients contributed.

```
pool_demo client 0 tame c0.dp 80      # ~1.3M DPs
pool_demo client 1 wild c1.dp 80      # ~1.2M DPs
pool_demo detect c0.dp c1.dp
  *** CROSS-CLIENT COLLISION ***
  tame DP from client 0  +  wild DP from client 1
  recovered k = 0x8000012345  (expected 0x8000012345)  -> MATCH
  => a fair split would credit clients {0, 1}
```

This is exactly the pool model: independent clients emit DPs against a **shared canonical
walk**; a central server detects the cross-client collision and attributes it. AMD + Nvidia
is just "more clients".

### Why a canonical spec is mandatory (the real interop blocker)

DPs from two clients only collide if **every client iterates the identical map**
`f(x) → jump[h(x)]`. Different jump-table size or points ⇒ different walk graph ⇒ herds from
different clients **never meet**. So the pool pins one **canonical spec** (in `pool_demo.hip`):

| Field | Value (demo) |
|---|---|
| range | `[0, 2^40)` (real: #135 uses `[2^134, 2^135)`) |
| `N_JUMPS` | 32 |
| jump scalars | deterministic LCG from fixed seed `0x1234567` (`canonical_jumps`) |
| `DP_BITS` | 12 (real: 34) |
| partition `h(x)` | `x.n[0] % N_JUMPS` |

Our HIP kangaroo and the Rust CPU solver in this repo currently use *different* tables
(HIP: 32/LCG, CPU: 256/FNV) — they are **not** interoperable until both adopt this spec.
RCKangaroo is a third; adapting its Nvidia fork to this canonical table is the next interop
task.

## Architecture (target)

```
AMD client (HIP + adapter) ─┐
Nvidia client (RCKangaroo   ├─ DP batches (TLS, canonical-spec-tagged)
  fork + sidecar) ──────────┘        │
                                     ▼
         DP INGEST / COORDINATION  (trust-minimized, auditable)
         • lease issuer: disjoint work, canonical spec, quotas
         • DP validator: mask + 1 scalar-mult endpoint check
         • sharded on-disk DP table (NEW build; dp_table.rs is a 64K toy)
         • global dedup + collision detector  ── winning (tame,wild) pair ─┐
         • signed Merkle contribution log (inclusion proofs)               │
                                                                           ▼
                           CUSTODY / PAYOUT  (t-of-n MPC — irreducible trust)
                           • quorum resolves k inside MPC, k never materialized
                           • signs ONE pre-agreed split tx (outputs = shares)
                           • broadcast via private relay
```

## Roadmap

- **P0 ✅** cross-client DP pooling + collision + attribution (`pool_demo.hip`).
- **P1** trust-minimized MVP: coordination server (leases + validate + dedup + detect),
  AMD adapter on `kangaroo.hip`, **RCKangaroo fork + sidecar**, Merkle contribution log.
  Custody = single operator (be explicit it's "trust the operator, verifiable accounting").
- **P2** anti-cheat: rate caps vs physical hashrate, challenge sub-ranges, stake slashing.
- **P3** distribute custody: t-of-n threshold-ECDSA / MPC (likely a fixed small committee).
- **P4** protected settlement: private-relay/miner submission, rehearsed on a **low-value
  exposed-pubkey puzzle on testnet** before ever pointing the swarm at #135.

> ⚠️ **Never fund a test challenge on mainnet.** A small-range key is solvable by anyone in
> minutes — real funds get swept. Test the full solve→split→payout flow on **testnet/regtest**.
