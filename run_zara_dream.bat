@echo off
:: Activate conda environment
call YOUR CONDA PATH AND ENV NAME HERE

:: Set API key
set OPENAI_API_KEY=YOUR API KEY HERE 
set REPLICATE_API_TOKEN=YOUR API KEY HERE 
set ELEVENLABS_API_KEY=YOUR API KEY HERE 


:: Navigate to project directory
D:
cd YOUR PROJECT DIRECTORY "

:: Run the dream, log the output with a timestamp
echo. >> dream_log.txt
echo ============================================================ >> dream_log.txt
echo Dream run: %DATE% %TIME% >> dream_log.txt
python CHARACTER_NAME_dream.py >> dream_log.txt 2>&1