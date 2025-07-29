#!/usr/bin/env python3
"""
Smoke Test for LoRA Adapter Fine-Tuning
CPU-only, using renewable_energy_dataset.json (~5k examples)
"""

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from lora_adapter_setup import LoRAAdapterSetup


def main():
    """Main function to execute the smoke test for fine-tuning."""
    print("🚀 Starting Smoke Test for LoRA Adapter Fine-Tuning")
    
    # Load LoRA adapter setup
    lora_setup = LoRAAdapterSetup()
    lora_setup.load_model_and_tokenizer()
    lora_setup.configure_lora()
    tokenized_dataset = lora_setup.prepare_dataset()
    
    # Define data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=lora_setup.tokenizer,
        mlm=False
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./lora_checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=3,  # modify this value pragmatically
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./lora_logs",
        report_to=[],  # Disable logging to wandb for a quick smoke test
        learning_rate=5e-5
    )
    
    # Setup Trainer
    trainer = Trainer(
        model=lora_setup.peft_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Training
    print("🔁 Training started. Expect this to take up to 8 hours...")
    trainer.train()
    
    # Save LoRA adapter
    print("🔗 Finalizing and saving the LoRA adapter...")
    lora_setup.save_adapter("./final_lora_adapter")
    
    print("✅ Smoke test complete! Adapter training was successfully completed overnight.")


if __name__ == "__main__":
    main()

