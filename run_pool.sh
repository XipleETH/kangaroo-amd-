#!/bin/bash
echo "============================================================"
echo "  Kangaroo ECDLP Solver - Pool Mode"
echo "  Collision Protocol (collisionprotocol.com)"
echo "  AMD GPU Optimized (RDNA3)"
echo "============================================================"
echo

# IMPORTANT: Change this to YOUR Bitcoin address!
WORKER="bc1qxtyjnyszrsvcwndvzsx6ee7s7hm5sg8uzl8duq"

# Puzzle #135
PUBKEY="02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
RANGE=135
START="40000000000000000000000000000000000"
KANGAROOS=65536
DP_BITS=28
DP_FILE="dp_output.bin"

echo "Worker:     $WORKER"
echo "Puzzle:     #135 (13.5 BTC)"
echo "[!] Make sure to change WORKER to YOUR Bitcoin address!"
echo

rm -f "$DP_FILE"

# Start bridge in background
python3 -u bridge/pool_bridge.py --worker "$WORKER" --dp-file "$DP_FILE" &
BRIDGE_PID=$!
sleep 3

# Start solver
./target/release/kangaroo --pubkey "$PUBKEY" --range "$RANGE" --start "$START" \
    --kangaroos "$KANGAROOS" --dp-bits "$DP_BITS" --mode wild --dp-output "$DP_FILE"

# Cleanup
kill $BRIDGE_PID 2>/dev/null
