@echo off
REM Self-contained HIP toolchain validation for the AMD RX 7800 XT (gfx1101).
REM Sets HIP + MSVC env, compiles hello.hip, runs it.

set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

echo === Loading MSVC (vcvars64) ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1 || ( echo MSVC cl.exe not found; check the Visual Studio path & exit /b 1 )

cd /d "%~dp0"

echo === Compiling hello.hip for gfx1101 ===
hipcc --offload-arch=gfx1101 hello.hip -o hello.exe
if errorlevel 1 ( echo COMPILE FAILED & exit /b 1 )

echo === Running hello.exe ===
hello.exe
