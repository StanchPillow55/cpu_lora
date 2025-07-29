#!/usr/bin/env python3
"""
Overnight Training Script for LoRA Adapter Fine-Tuning
CPU-optimized with monitoring and time estimation
Target: ~8 hours maximum training time
"""

import torch
import time
import transformers
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.8.0"), "This script requires transformers >= 4.8.0"
import os
import json
from datetime import datetime, timedelta
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import wandb

class TimeMonitoringCallback(TrainerCallback):
    """Custom callback to monitor training time and stop if approaching time limit."""
    
    def __init__(self, max_hours=8):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600
        
    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            print(f"⏰ Time limit reached ({elapsed/3600:.1f}h). Stopping training...")
            control.should_training_stop = True
            
    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        remaining = self.max_seconds - elapsed
        print(f"📊 Epoch completed. Elapsed: {elapsed/3600:.1f}h, Remaining: {remaining/3600:.1f}h")

def estimate_training_time(dataset_size, batch_size, epochs, steps_per_second=0.5):
    """Estimate training time based on dataset size and hardware."""
    total_steps = (dataset_size // batch_size) * epochs
    estimated_seconds = total_steps / steps_per_second
    return estimated_seconds / 3600  # Convert to hours

def setup_overnight_training():
    """Setup and execute overnight training."""
    print("🌙 Overnight LoRA Fine-Tuning Setup")
    print("=" * 40)
    
    # Initialize wandb for monitoring
    wandb.init(
        project="cpu-lora-renewable-energy",
        name=f"overnight-training-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "lora_rank": 8,
            "lora_alpha": 16,
            "learning_rate": 5e-5,
            "batch_size": 2,  # Reduced for CPU efficiency
            "max_length": 512,
            "device": "cpu"
        }
    )
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"🔄 Loading {model_name}...")
    
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
    print("🔧 Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Load and prepare dataset
    print("📊 Loading dataset...")
    with open("renewable_energy_dataset.json", 'r') as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        texts = []
        for instruction, output in zip(examples['instruction'], examples['output']):
            text = f"Question: {instruction}\nAnswer: {output}{tokenizer.eos_token}"
            texts.append(text)
            
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Estimate training time
    dataset_size = len(tokenized_dataset)
    batch_size = 2
    epochs = 1  # Start with 1 epoch for overnight test
    
    estimated_hours = estimate_training_time(dataset_size, batch_size, epochs)
    print(f"⏱️  Estimated training time: {estimated_hours:.1f} hours")
    
    if estimated_hours > 8:
        print("⚠️  Warning: Estimated time exceeds 8 hours!")
        epochs = max(1, int(8 / estimated_hours))
        print(f"🔄 Adjusting to {epochs} epoch(s)")
    
    # Setup training arguments for CPU optimization
    training_args = TrainingArguments(
        output_dir="./overnight_lora_checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        dataloader_num_workers=0,  # Important for CPU training
        fp16=False,  # Disable for CPU
        bf16=False,  # Disable for CPU
        dataloader_pin_memory=False,  # Disable for CPU
        report_to=["wandb", "tensorboard"],
        logging_dir="./overnight_logs",
        run_name=f"overnight-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup trainer with time monitoring
    time_monitor = TimeMonitoringCallback(max_hours=8)
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        callbacks=[time_monitor]
    )
    
    # Start training
    start_time = datetime.now()
    print(f"🚀 Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Expected completion: {(start_time + timedelta(hours=estimated_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("💤 Starting overnight training...")
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
    
    # Save the final adapter
    final_output_dir = "./overnight_trained_adapter"
    os.makedirs(final_output_dir, exist_ok=True)
    peft_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print(f"\n🏁 Training Summary:")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {training_duration}")
    print(f"   Adapter saved to: {final_output_dir}")
    
    # Finish wandb run
    wandb.finish()
    
    return final_output_dir

if __name__ == "__main__":
    # Set CPU-specific environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "8"  # Adjust based on your CPU cores
    
    # Run overnight training
    adapter_path = setup_overnight_training()
    print(f"🎉 Overnight training complete! Adapter available at: {adapter_path}")
