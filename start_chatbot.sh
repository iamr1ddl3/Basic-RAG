#!/bin/bash

echo "Starting RAG Chatbot..."

# Run the startup script
python start_chatbot.py

# Check if script exited with an error
if [ $? -ne 0 ]; then
    echo "Startup failed. Please check the logs."
    exit 1
fi 