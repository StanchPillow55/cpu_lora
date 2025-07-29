#!/usr/bin/env python3
"""
LoRA Adapter Setup Smoke Test
Isolate and verify LoRA configuration and loading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def run_lora_smoke_test():
    """Run smoke test for LoRA setup."""
    print("🔄 Running LoRA setup smoke test...")
    
    # Test configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        return False
    
    # Configure LoRA adapter
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        print("✅ LoRA adapter configured successfully")
    except Exception as e:
        print(f"❌ Error configuring LoRA adapter: {e}")
        return False
    
    # Test LoRA model loading
    try:
        lora_model.eval()  # Set to evaluation mode for verification
        print("✅ LoRA model is ready for use")
    except Exception as e:
        print(f"❌ Error with LoRA model: {e}")
        return False

    return True

if __name__ == "__main__":
    if run_lora_smoke_test():
        print("\n🎉 LoRA setup smoke test passed!")
    else:
        print("\n❌ LoRA setup smoke test failed!")

