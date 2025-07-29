#!/usr/bin/env python3
"""
LoRA Adapter Setup for CPU-Optimized Fine-Tuning
TinyLlama-1.1B with rank=8, alpha=16, CPU-only
"""

import torch
import transformers
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.8.0"), "This script requires transformers >= 4.8.0"
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import json
from datasets import Dataset
from typing import Dict, List
import os

class LoRAAdapterSetup:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"🔄 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with CPU-optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded successfully on {self.device}")
        
    def configure_lora(self):
        """Configure LoRA adapter with specified parameters."""
        print("🔧 Configuring LoRA adapter...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=8,                              # LoRA rank
            lora_alpha=16,                    # LoRA alpha
            target_modules=["q_proj", "v_proj"],  # Target attention modules
            lora_dropout=0.05,                # Dropout rate
            bias="none",                      # No bias training
            task_type=TaskType.CAUSAL_LM,     # Causal language modeling
        )
        
        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        print("✅ LoRA adapter configured successfully")
        return lora_config
        
    def prepare_dataset(self, dataset_path: str = "renewable_energy_dataset.json"):
        """Prepare dataset for training."""
        print(f"📊 Loading dataset from {dataset_path}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_list(data)
        
        def tokenize_function(examples):
            # Combine instruction and output for causal LM
            texts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                text = f"Question: {instruction}\nAnswer: {output}{self.tokenizer.eos_token}"
                texts.append(text)
                
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
            
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
        
    def save_adapter(self, output_dir: str = "./lora_adapter"):
        """Save the LoRA adapter."""
        if self.peft_model is None:
            raise ValueError("LoRA model not configured. Run configure_lora() first.")
            
        os.makedirs(output_dir, exist_ok=True)
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ LoRA adapter saved to {output_dir}")
        
    def load_adapter(self, adapter_path: str = "./lora_adapter"):
        """Load a previously saved LoRA adapter."""
        print(f"🔄 Loading LoRA adapter from {adapter_path}")
        
        # Load base model first if not loaded
        if self.model is None:
            self.load_model_and_tokenizer()
            
        # Load the PEFT model
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        
        print("✅ LoRA adapter loaded successfully")

def main():
    """Main function to demonstrate LoRA setup."""
    print("🚀 LoRA Adapter Setup for CPU-Optimized Fine-Tuning")
    print("=" * 55)
    
    # Initialize setup
    lora_setup = LoRAAdapterSetup()
    
    # Load model and tokenizer
    lora_setup.load_model_and_tokenizer()
    
    # Configure LoRA
    lora_config = lora_setup.configure_lora()
    
    # Display configuration
    print("\n📋 LoRA Configuration:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha (α): {lora_config.lora_alpha}")
    print(f"  - Target modules: {lora_config.target_modules}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - Task type: {lora_config.task_type}")
    
    # Save initial adapter
    lora_setup.save_adapter("./initial_lora_adapter")
    
    print("\n✅ LoRA setup complete!")
    print("🎯 Ready for training with the configured adapter.")

if __name__ == "__main__":
    main()
