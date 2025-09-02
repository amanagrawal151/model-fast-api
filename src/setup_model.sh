#!/bin/bash
# Only generate model if not already present
if [ ! -d "src/pii_bert" ]; then
    echo "Generating model artifacts..."
    python src/model.py
fi