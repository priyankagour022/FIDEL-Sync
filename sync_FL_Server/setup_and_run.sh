#!/bin/bash

# Create virtual environment
python3 -m venv fl

# Activate virtual environment
source fl/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run your Python script
python server.py


#chmod +x setup_and_run.sh