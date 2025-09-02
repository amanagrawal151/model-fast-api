#!/bin/bash
# Only generate model if not already present
if [ ! -d "pii_bert" ]; then
    echo "Generating model artifacts..."
    python model.py
fi