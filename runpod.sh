#!/bin/bash

apt update -y

apt install nano screen htop -y

# Create a new screen session
screen -dmS benchmark_session

# Install common dependencies
screen -S benchmark_session -X stuff "pip install -r requirements.txt\n"

# Run the main script
screen -S benchmark_session -X stuff "python main.py\n"

#screen -S benchmark_session -X stuff "if [ \"$DEBUG\" == \"False\" ]; then runpodctl remove pod \$RUNPOD_POD_ID; fi\n"

/start.sh