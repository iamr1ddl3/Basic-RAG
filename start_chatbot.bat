@echo off
echo Starting RAG Chatbot...

:: Run the startup script
python start_chatbot.py

:: Check if script exited with an error
if %errorlevel% neq 0 (
    echo Startup failed. Please check the logs.
    pause
    exit /b %errorlevel%
)

:: Keep console open if script exits
pause 