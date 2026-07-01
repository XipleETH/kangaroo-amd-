@echo off
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
hipcc -O3 --offload-arch=gfx1101 ec_test.hip -o ec_test.exe 1>ec_compile.txt 2>&1
echo HIPCC_EXIT=%errorlevel% >>ec_compile.txt
