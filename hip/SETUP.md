# Stage 1 — HIP/ROCm toolchain setup (AMD RX 7800 XT, Windows 10)

Goal: get native HIP compiling and running on your gfx1101 GPU, validated with a
trivial kernel, **before** we write any secp256k1 code.

## Feasibility (already confirmed, July 2026)
- RX 7800 XT = **gfx1101**, officially supported for HIP **compute runtime** on Windows.
- **Windows 10 64-bit is supported** by HIP SDK for Windows 7.1.
- You already have Visual Studio 2022 (MSVC) — hipcc uses it as the host compiler.

## Install (needs admin + one reboot, ~1 hour, a few GB)

1. **AMD Adrenalin driver** with HIP/ROCm support — get the latest for the RX 7800 XT:
   https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-7000-series/amd-radeon-rx-7800-xt.html
   (A clean/factory-reset install is safest. Reboot after.)

2. **AMD HIP SDK for Windows 7.1.1**:
   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
   Run the installer as admin; select the **HIP SDK** (compiler + runtime + math libs).
   It sets `HIP_PATH` and usually adds a "HIP SDK Command Prompt" to the Start menu.

3. **Reboot.**

## Validate (this is the go/no-go gate)

Open a terminal that can see `hipcc` (the "HIP SDK Command Prompt", or any terminal
with `%HIP_PATH%\bin` on PATH), `cd` into this folder, and run:

```
build_hello.bat
```

**PASS looks like:**
```
Device:         AMD Radeon RX 7800 XT
gcnArchName:    gfx1101
...
GPU COMPUTE OK  -> toolchain validated, go to Stage 2
```

- If it prints **"HIP devices found: 0"** or only shows a CPU → driver/SDK mismatch;
  the GPU isn't enumerated. Don't proceed — we fix the install first.
- If `hipcc` isn't found → use the HIP SDK Command Prompt or fix PATH.
- Expect a slow first compile (AMD first-compile latency; normal, not a hang).

## Then: Stage 2 — the port
Once the gate passes, we port RCKangaroo's field arithmetic to HIP (with AMD carry
emulation), then the EC ops and the kangaroo loop, validating each against a CPU
reference on small ranges — same discipline as the WGSL work.
