@echo off
REM ============================================================================
REM  AMD Kangaroo -> Collision Protocol Pool Runner
REM  Puzzle #135 - 13.5 BTC Prize
REM ============================================================================

set KANGAROO=d:\hacks\bruteforce\tools\kangaroo\target\release\kangaroo.exe
set BRIDGE=d:\hacks\bruteforce\tools\kangaroo\bridge\pool_bridge.py
set WORKER=bc1qxtyjnyszrsvcwndvzsx6ee7s7hm5sg8uzl8duq
set DP_FILE=d:\hacks\bruteforce\tools\kangaroo\dp_output.bin

echo ============================================================
echo   AMD Kangaroo Pool Mode - Puzzle #135
echo   Worker: %WORKER%
echo ============================================================

REM Delete old DP file
if exist "%DP_FILE%" del "%DP_FILE%"

REM Start the bridge in background
echo [1] Starting pool bridge...
start "Pool Bridge" cmd /c "python -u -X utf8 %BRIDGE% --worker %WORKER% --dp-file %DP_FILE%"

REM Wait for bridge to connect and get work assignment
timeout /t 5 /nobreak > nul

REM Start the solver
echo [2] Starting AMD GPU solver...
%KANGAROO% --pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 --range 135 --kangaroos 65536 --dp-bits 28 --mode wild --dp-output "%DP_FILE%"

pause
