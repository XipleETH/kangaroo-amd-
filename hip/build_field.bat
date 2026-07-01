@echo off
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
echo === Compiling field_test.hip for gfx1101 ===
hipcc -O3 --offload-arch=gfx1101 field_test.hip -o field_test.exe
if errorlevel 1 ( echo COMPILE FAILED & exit /b 1 )
echo === Running ===
field_test.exe
