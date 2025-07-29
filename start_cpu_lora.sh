#!/bin/bash

# CPU-Optimized LoRA Fine-Tuning Quick Start Script

echo "🚀 Starting CPU-Optimized LoRA Fine-Tuning Environment..."

# Activate virtual environment
echo "📦 Activating virtual environment (cpu_lora_env)..."
source cpu_lora_env/bin/activate

# Check if activation was successful
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Start Jupyter Notebook
echo "📓 Starting Jupyter Notebook..."
echo "Available notebooks:"
echo "  - validate_setup.ipynb         : Validate library installations"
echo "  - peft_lora_setup.ipynb       : PEFT with transformers setup"
echo "  - quantization_verification.ipynb : llama-cpp-python verification"
echo "  - nf4_quantization_test.ipynb : NF-4 quantization testing"
echo ""
echo "🌐 Opening Jupyter Notebook in your browser..."

jupyter notebook
