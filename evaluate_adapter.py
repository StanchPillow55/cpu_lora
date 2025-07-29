#!/usr/bin/env python3
"""
Evaluation Script for Fine-tuned LoRA Adapter
Post-training sanity check with AlpacaEval and basic metrics
"""

import torch
import json
import transformers
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.8.0"), "This script requires transformers >= 4.8.0"
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import numpy as np
from typing import List, Dict
import time

class AdapterEvaluator:
    def __init__(self, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.base_model_name = base_model_name
        self.base_model = None
        self.fine_tuned_model = None
        self.tokenizer = None
        
    def load_models(self, adapter_path: str = "./overnight_trained_adapter"):
        """Load both base and fine-tuned models for comparison."""
        print("🔄 Loading models for evaluation...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load base model
        print("   Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Load fine-tuned model
        print("   Loading fine-tuned model...")
        if os.path.exists(adapter_path):
            self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            print("✅ Models loaded successfully")
        else:
            print(f"❌ Adapter not found at {adapter_path}")
            self.fine_tuned_model = self.base_model
            
    def generate_response(self, model, prompt: str, max_length: int = 200) -> str:
        """Generate response from a model."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        response = response[len(prompt):].strip()
        return response
        
    def evaluate_on_test_questions(self, test_questions: List[str]) -> Dict:
        """Evaluate both models on test questions."""
        results = {
            "base_model_responses": [],
            "fine_tuned_responses": [],
            "questions": test_questions
        }
        
        print("🧪 Evaluating models on test questions...")
        
        for i, question in enumerate(test_questions):
            print(f"   Question {i+1}/{len(test_questions)}: {question[:50]}...")
            
            # Format question as in training
            formatted_question = f"Question: {question}\nAnswer:"
            
            # Get base model response
            base_response = self.generate_response(self.base_model, formatted_question)
            results["base_model_responses"].append(base_response)
            
            # Get fine-tuned model response
            ft_response = self.generate_response(self.fine_tuned_model, formatted_question)
            results["fine_tuned_responses"].append(ft_response)
            
        return results
        
    def calculate_perplexity(self, model, text_samples: List[str]) -> float:
        """Calculate average perplexity on text samples."""
        print("📊 Calculating perplexity...")
        
        perplexities = []
        
        for text in text_samples[:10]:  # Limit to 10 samples for speed
            inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                
        avg_perplexity = np.mean(perplexities)
        return avg_perplexity
        
    def run_basic_alpaca_eval(self, results: Dict) -> Dict:
        """Run basic AlpacaEval-style evaluation."""
        print("🦙 Running basic AlpacaEval-style evaluation...")
        
        evaluation_metrics = {
            "response_lengths": {
                "base_model": [],
                "fine_tuned": []
            },
            "contains_domain_keywords": {
                "base_model": 0,
                "fine_tuned": 0
            }
        }
        
        # Domain-specific keywords for renewable energy
        domain_keywords = [
            "solar", "wind", "renewable", "energy", "power", "electricity",
            "battery", "storage", "hydro", "geothermal", "turbine", "panel",
            "sustainable", "carbon", "emission", "fossil", "fuel"
        ]
        
        for base_resp, ft_resp in zip(results["base_model_responses"], results["fine_tuned_responses"]):
            # Response length analysis
            evaluation_metrics["response_lengths"]["base_model"].append(len(base_resp.split()))
            evaluation_metrics["response_lengths"]["fine_tuned"].append(len(ft_resp.split()))
            
            # Domain keyword analysis
            base_keywords = sum(1 for keyword in domain_keywords if keyword.lower() in base_resp.lower())
            ft_keywords = sum(1 for keyword in domain_keywords if keyword.lower() in ft_resp.lower())
            
            evaluation_metrics["contains_domain_keywords"]["base_model"] += base_keywords
            evaluation_metrics["contains_domain_keywords"]["fine_tuned"] += ft_keywords
            
        # Calculate averages
        evaluation_metrics["avg_response_length"] = {
            "base_model": np.mean(evaluation_metrics["response_lengths"]["base_model"]),
            "fine_tuned": np.mean(evaluation_metrics["response_lengths"]["fine_tuned"])
        }
        
        return evaluation_metrics
        
    def generate_evaluation_report(self, results: Dict, metrics: Dict, perplexity: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("🔍 LORA ADAPTER EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Perplexity comparison
        report.append("📊 PERPLEXITY ANALYSIS:")
        report.append(f"   Base Model Perplexity: {perplexity['base_model']:.2f}")
        report.append(f"   Fine-tuned Perplexity: {perplexity['fine_tuned']:.2f}")
        
        improvement = perplexity['base_model'] - perplexity['fine_tuned']
        if improvement > 0:
            report.append(f"   ✅ Improvement: {improvement:.2f} (Lower is better)")
        else:
            report.append(f"   ⚠️  Regression: {abs(improvement):.2f}")
        report.append("")
        
        # Response length analysis
        report.append("📝 RESPONSE LENGTH ANALYSIS:")
        report.append(f"   Base Model Avg Length: {metrics['avg_response_length']['base_model']:.1f} words")
        report.append(f"   Fine-tuned Avg Length: {metrics['avg_response_length']['fine_tuned']:.1f} words")
        report.append("")
        
        # Domain knowledge analysis
        report.append("🎯 DOMAIN KNOWLEDGE ANALYSIS:")
        report.append(f"   Base Model Domain Keywords: {metrics['contains_domain_keywords']['base_model']}")
        report.append(f"   Fine-tuned Domain Keywords: {metrics['contains_domain_keywords']['fine_tuned']}")
        
        keyword_improvement = metrics['contains_domain_keywords']['fine_tuned'] - metrics['contains_domain_keywords']['base_model']
        if keyword_improvement > 0:
            report.append(f"   ✅ Improvement: +{keyword_improvement} domain-specific terms")
        else:
            report.append(f"   ⚠️  No improvement in domain specificity")
        report.append("")
        
        # Sample comparisons
        report.append("💬 SAMPLE RESPONSE COMPARISONS:")
        for i, (question, base_resp, ft_resp) in enumerate(zip(
            results['questions'][:3], 
            results['base_model_responses'][:3], 
            results['fine_tuned_responses'][:3]
        )):
            report.append(f"\n   Question {i+1}: {question}")
            report.append(f"   Base Model: {base_resp[:100]}...")
            report.append(f"   Fine-tuned: {ft_resp[:100]}...")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main evaluation function."""
    print("🔍 Starting LoRA Adapter Evaluation")
    print("=" * 40)
    
    # Test questions for evaluation
    test_questions = [
        "What is renewable energy?",
        "How do solar panels work?",
        "What are the benefits of wind energy?",
        "How is hydroelectric power generated?",
        "What is the role of battery storage in renewable energy systems?",
        "What factors affect solar panel efficiency?",
        "How does geothermal energy work?",
        "What are the environmental benefits of renewable energy?"
    ]
    
    # Initialize evaluator
    evaluator = AdapterEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Run evaluation
    results = evaluator.evaluate_on_test_questions(test_questions)
    
    # Calculate perplexity
    sample_texts = [f"Question: {q}\nAnswer: Sample answer about renewable energy." for q in test_questions]
    base_perplexity = evaluator.calculate_perplexity(evaluator.base_model, sample_texts)
    ft_perplexity = evaluator.calculate_perplexity(evaluator.fine_tuned_model, sample_texts)
    
    perplexity_results = {
        "base_model": base_perplexity,
        "fine_tuned": ft_perplexity
    }
    
    # Run basic AlpacaEval
    metrics = evaluator.run_basic_alpaca_eval(results)
    
    # Generate and save report
    report = evaluator.generate_evaluation_report(results, metrics, perplexity_results)
    
    # Save detailed results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "results": results,
            "metrics": metrics,
            "perplexity": perplexity_results
        }, f, indent=2)
    
    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    print("\n✅ Evaluation complete!")
    print("📄 Detailed results saved to: evaluation_results.json")
    print("📄 Report saved to: evaluation_report.txt")

if __name__ == "__main__":
    main()
