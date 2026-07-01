@echo off
REM Generic HIP build: mk.bat <source.hip>  -> <source>.exe  (compile log in <source>_compile.txt)
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
hipcc -O3 --offload-arch=gfx1101 %1 -o %~n1.exe 1>%~n1_compile.txt 2>&1
echo HIPCC_EXIT=%errorlevel% >>%~n1_compile.txt
