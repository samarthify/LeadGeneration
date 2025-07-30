#!/bin/bash

# Build script for Vercel deployment
echo "Starting build process..."

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies with verbose output
echo "Installing Python dependencies..."
pip install -r requirements.txt --verbose

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
else
    echo "Build failed!"
    exit 1
fi 