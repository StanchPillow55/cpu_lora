#!/usr/bin/env python3
"""
Training Smoke Test
Isolate and verify training setup with minimal steps
"""

import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

def run_training_smoke_test():
    """Run a minimal training smoke test."""
    print("🔄 Running training smoke test...")
    
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        # Load model and tokenizer
        print("   Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Configure LoRA
        print("   Configuring LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16, 
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        peft_model = get_peft_model(model, lora_config)
        
        # Load dataset (just first 10 examples for smoke test)
        print("   Loading small dataset sample...")
        with open("renewable_energy_dataset.json", 'r') as f:
            data = json.load(f)
        
        # Use only first 10 examples for quick test
        small_data = data[:10]
        dataset = Dataset.from_list(small_data)
        
        def tokenize_function(examples):
            texts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                text = f"Question: {instruction}\nAnswer: {output}{tokenizer.eos_token}"
                texts.append(text)
                
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,  # Shorter for smoke test
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Setup minimal training arguments
        print("   Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./smoke_test_checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Just 1 epoch
            per_device_train_batch_size=1,  # Smallest batch size
            max_steps=2,  # Only 2 steps for smoke test
            logging_steps=1,
            save_steps=999999,  # Don't save checkpoints
            dataloader_num_workers=0,
            fp16=False,
            bf16=False,
            dataloader_pin_memory=False,
            report_to=[],  # No reporting for smoke test
            logging_dir=None
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Setup trainer
        print("   Setting up trainer...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset
        )
        
        # Run minimal training
        print("   Running 2 training steps...")
        trainer.train()
        
        print("✅ Training smoke test completed successfully")
        
        # Clean up
        if os.path.exists("./smoke_test_checkpoints"):
            import shutil
            shutil.rmtree("./smoke_test_checkpoints")
            print("   Cleaned up test checkpoints")
        
        return True
        
    except Exception as e:
        print(f"❌ Training smoke test failed: {e}")
        return False

if __name__ == "__main__":
    if run_training_smoke_test():
        print("\n🎉 Training smoke test passed!")
    else:
        print("\n❌ Training smoke test failed!")
