@echo off
echo ============================================================
echo   CHARACTER NAME Speaks V3 — Environment Startup
echo ============================================================

:: Activate conda environment
call ### YOUR PATH TO CONDA AND ENV NAME HERE

:: Set API key
set OPENAI_API_KEY=YOUR API KEY HERE
set REPLICATE_API_TOKEN=YOUR API KEY HERE
set ELEVENLABS_API_KEY=YOUR API KEY HERE


:: Navigate to project directory
D:
cd "YOUR project directory here"

echo.
echo   Environment: CUSTOMIZED NAME (Python 3.11)
echo   OPENAI_API_KEY: set
echo   Directory: %CD%
echo.
echo   Ready! Run:  python YOUR PROGRAM NAME HERE.py
echo ============================================================

:: Keep the terminal open for interactive use
cmd /k
