#!/bin/bash
# Only download and extract if not already present
if [ ! -d "pii_bert" ]; then
    echo "Downloading model artifacts from Google Drive..."
    # Use gdown to download from Google Drive
    pip install gdown
    gdown --id 1wKNCdK-GqRU9Dl8eJ91iI99A0muR6iGW -O train.zip
    unzip train.zip
    rm train.zip
fi