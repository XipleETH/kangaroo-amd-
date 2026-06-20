#!/bin/bash
echo "============================================================"
echo "  Kangaroo ECDLP Solver - Local Solo Mode"
echo "  AMD GPU Optimized (RDNA3)"
echo "============================================================"
echo

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_local.sh <pubkey> <range_bits> [dp_bits] [kangaroos]"
    echo
    echo "Example - Bitcoin Puzzle #40:"
    echo "  ./run_local.sh 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 40 10 65536"
    exit 1
fi

PUBKEY=$1
RANGE=$2
DP_BITS=${3:-20}
KANGAROOS=${4:-65536}

echo "Target:     $PUBKEY"
echo "Range:      $RANGE bits"
echo "DP bits:    $DP_BITS"
echo "Kangaroos:  $KANGAROOS"
echo

./target/release/kangaroo --pubkey "$PUBKEY" --range "$RANGE" --dp-bits "$DP_BITS" --kangaroos "$KANGAROOS"
