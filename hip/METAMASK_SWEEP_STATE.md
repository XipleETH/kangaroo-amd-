# MetaMask sweeper — state & handoff (read this to resume, e.g. on Ubuntu)

**To resume:** open Claude Code in this repo and say:
> "Continue the Guntis MetaMask sweeper — read tools/kangaroo/hip/METAMASK_SWEEP_STATE.md"

## Goal

Crack the **Guntis Vitolins 8.6 ETH** MetaMask challenge (floflo777/open-crypto-puzzles).
- Target ETH address: `0x9c2f44efad0c1e852a09df9939e6daf061140caf`
- Mechanism (certified): 12-word BIP39 mnemonic, no passphrase → MetaMask path
  `m/44'/60'/0'/0/0` → keccak(pubkey) → address.
- The one **untested** lead floflo priced but couldn't run without a GPU: the **substring
  word pool** sweep (~2.78e11 derivations). Mechanism is UNCONFIRMED (may yield nothing).

## DONE — full GPU crypto pipeline, built from scratch in HIP, all validated

| File | What | Validated against |
|---|---|---|
| `sha.h` | SHA-256, SHA-512 (sliding 16-word schedule; word-level absorb + finalize) | 7 FIPS vectors (`sha_test.hip`) |
| `bip39.h` | HMAC-SHA512 (ipad/opad midstate), PBKDF2 seed (u64 hot loop) | RFC 4231, BIP39 seed vector (`bip39_test.hip`) |
| `keccak.h` | Keccak-256 (Ethereum 0x01 padding) | 2 vectors (`bip32_test.hip`) |
| `bip32.h` | BIP32 derive `m/44'/60'/0'/0/0`, mod-n add, priv→pubkey, ETH address; reuses `secp256k1.h`+`ec.h` | end-to-end reproduces `0x9858...` (`bip32_test.hip`) |
| `metamask_bench.hip` | throughput benchmark (full pipeline + PBKDF2-only) | — |

**`bip32_test.exe` prints "FULL BIP39->ETH PIPELINE VALIDATED".** The whole chain is correct.

## Speed (Windows, RX 7800 XT)

- Full derivation: **~225k/s**.  PBKDF2-only: ~300k/s. Occupancy-bound (SHA-512 register pressure).
- floflo got ~792k/s (better/other GPU). **Expect a meaningful bump on Linux/ROCm** (mature
  compiler + no 2s TDR). Full substring sweep at 225k/s ≈ 14 days; on Linux/a bigger GPU, less.

## NEXT STEPS (not built yet)

1. **Enumerator + BIP39 checksum filter** (the sweep kernel):
   - map a linear index → 12 word indices under the constraints below;
   - pack entropy, check BIP39 checksum via SHA-256 (only ~1/16 pass → then PBKDF2+derive);
   - build the mnemonic string, run the validated pipeline, compare to the target.
   - **Validate it by planting a known witness** (a mnemonic in the swept set) and recovering it,
     the same protocol floflo used.
2. **Staged / prioritized sweep**: test the most likely subspaces first (few substrings, higher-
   likelihood word combos) so a shallow answer surfaces in hours, not the full 14 days.
3. Optional: push kernel speed (split PBKDF2 kernel from derive kernel for occupancy; LDS for
   ipad/opad; check VGPR count with `--save-temps`).

## The word pool (from `analysis/` of the puzzle; verify against BIP39 wordlist)

Anchors: pos1=`dutch`, pos5∈{`fog`,`cloud`}, pos12=`parrot`. Confirmed members: `fiber`, `fork`.
Partition: 6 words from the VIDEO side, 6 from the POST side.

5 planted sentences → BIP39 words + substrings:
- V1 "Don't expect anything easy there will be dark fog on the lake": full=expect,easy,there,will,fog,lake; sub=any,ill,thing
- V2 "Do you think its more likely for parrot can sing a song then for a goat to whistle": full=you,more,parrot,can,sing,song,then,goat; sub=hen,like
- A1 "Round dutch cattle is living in the forest and eating wood": full=round,dutch,cattle,forest,wood; sub=cat
- A2 "Only because there is a lot of healthy fiber": full=only,because,there,fiber; sub=cause,health,use
- A3 "Hunter like the rib roast dinner fresh": full=like,rib,roast,dinner,fresh; sub=hunt,inner
- Video metadata: top,update,winter,finish.  Post tags: season,market,fork,round.

floflo already tested (0 match): all full words + inflections + metadata, both 6/6 and free split,
~16.75B derivations. The UNTESTED addition is the **substrings** above.

## Build

Windows: `mk.bat <file>.hip` (loads vcvars + `hipcc -O3 --offload-arch=gfx1101`).
Linux/ROCm: `make tests` (see `Makefile`). Requires ROCm + `hipcc` on PATH; gfx1101 is supported.
For NVIDIA: `hipcc` targets CUDA automatically, or translate with `hipify`.
