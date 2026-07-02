@echo off
REM msweep.bat <RANGE_BITS> [DP_BITS]  -> builds kangaroo.exe for that range
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
if "%2"=="" ( set DPDEF= ) else ( set DPDEF=-DDP_BITS=%2 )
hipcc -O3 --offload-arch=gfx1101 -DRANGE_BITS=%1 %DPDEF% kangaroo.hip -o kangaroo.exe 1>sweep_compile.txt 2>&1
echo EXIT=%errorlevel% >>sweep_compile.txt
