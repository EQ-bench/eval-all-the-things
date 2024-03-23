#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py

if [ "$DEBUG" == "False" ]; then
	runpodctl remove pod $RUNPOD_POD_ID
fi