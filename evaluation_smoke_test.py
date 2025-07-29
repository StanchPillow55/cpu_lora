#!/usr/bin/env python3
"""
Evaluation Smoke Test
Isolate and verify evaluation setup without needing a trained adapter
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

def run_evaluation_smoke_test():
    """Run a minimal evaluation smoke test."""
    print("🔄 Running evaluation smoke test...")
    
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
        
        # Create a simple LoRA model for testing
        print("   Creating LoRA model for testing...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        lora_model = get_peft_model(model, lora_config)
        
        # Test text generation
        print("   Testing text generation...")
        test_prompt = "Question: What is renewable energy?\\nAnswer:"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 20,  # Short generation
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated response: {response[:100]}...")
        
        # Test perplexity calculation
        print("   Testing perplexity calculation...")
        test_text = "Renewable energy is sustainable power from natural sources."
        inputs = tokenizer.encode(test_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            print(f"   Sample perplexity: {perplexity.item():.2f}")
        
        # Test domain keyword analysis
        print("   Testing domain keyword analysis...")
        domain_keywords = ["solar", "wind", "renewable", "energy", "power"]
        test_response = "Solar panels and wind turbines generate renewable energy."
        
        keyword_count = sum(1 for keyword in domain_keywords if keyword.lower() in test_response.lower())
        print(f"   Found {keyword_count} domain keywords in test response")
        
        print("✅ Evaluation smoke test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation smoke test failed: {e}")
        return False

if __name__ == "__main__":
    if run_evaluation_smoke_test():
        print("\\n🎉 Evaluation smoke test passed!")
    else:
        print("\\n❌ Evaluation smoke test failed!")
