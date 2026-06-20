@echo off
echo ============================================================
echo   Kangaroo ECDLP Solver - Pool Mode
echo   Collision Protocol (collisionprotocol.com)
echo   AMD GPU Optimized (RDNA3)
echo ============================================================
echo.

REM ============================================================
REM  IMPORTANT: Change this to YOUR Bitcoin address!
REM  This is where pool rewards will be sent.
REM  Default address below is for donations to the fork author.
REM ============================================================
set WORKER=bc1qxtyjnyszrsvcwndvzsx6ee7s7hm5sg8uzl8duq

REM Puzzle #135 configuration
set PUBKEY=02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
set RANGE=135
set START=40000000000000000000000000000000000
set KANGAROOS=65536
set DP_BITS=28
set DP_FILE=dp_output.bin

echo Worker:     %WORKER%
echo Puzzle:     #135 (13.5 BTC)
echo DP bits:    %DP_BITS%
echo Kangaroos:  %KANGAROOS%
echo.
echo [!] Make sure to change WORKER to YOUR Bitcoin address!
echo.

REM Clean old DP file
if exist %DP_FILE% del %DP_FILE%

REM Start pool bridge in background
echo Starting pool bridge...
start /b python -u bridge/pool_bridge.py --worker %WORKER% --dp-file %DP_FILE%

REM Wait for bridge to connect
timeout /t 3 /nobreak >nul

REM Start solver
echo Starting GPU solver...
target\release\kangaroo.exe --pubkey %PUBKEY% --range %RANGE% --start %START% --kangaroos %KANGAROOS% --dp-bits %DP_BITS% --mode wild --dp-output %DP_FILE%
