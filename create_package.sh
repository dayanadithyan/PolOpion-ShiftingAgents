#!/bin/bash

# Create main directory
mkdir political_opinion_analysis
cd political_opinion_analysis

# Create Python package files and directories
touch __init__.py
touch agent.py
touch utils.py
mkdir models
touch models/__init__.py
touch models/sentiment_analysis_model.pth
touch models/relevance_analysis_model.pth
mkdir data
mkdir tests

# Create a sample test file
touch tests/test_agent.py

# Output success message
echo "Political opinion analysis package created successfully!"
