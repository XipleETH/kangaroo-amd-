@echo off
echo ============================================================
echo   Kangaroo ECDLP Solver - Local Solo Mode
echo   AMD GPU Optimized (RDNA3)
echo ============================================================
echo.

if "%~1"=="" (
    echo Usage: run_local.bat ^<pubkey^> ^<range_bits^> [dp_bits] [kangaroos]
    echo.
    echo Example - Bitcoin Puzzle #40:
    echo   run_local.bat 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 40 10 65536
    echo.
    echo Example - Bitcoin Puzzle #135:
    echo   run_local.bat 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 135 24 65536
    echo.
    exit /b 1
)

set PUBKEY=%~1
set RANGE=%~2
set DP_BITS=%~3
set KANGAROOS=%~4

if "%DP_BITS%"=="" set DP_BITS=20
if "%KANGAROOS%"=="" set KANGAROOS=65536

echo Target:     %PUBKEY%
echo Range:      %RANGE% bits
echo DP bits:    %DP_BITS%
echo Kangaroos:  %KANGAROOS%
echo.
echo Starting solver...
echo.

target\release\kangaroo.exe --pubkey %PUBKEY% --range %RANGE% --dp-bits %DP_BITS% --kangaroos %KANGAROOS%
