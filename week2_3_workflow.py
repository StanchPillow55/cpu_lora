#!/usr/bin/env python3
"""
Week 2-3 Workflow: Adapter Design and Training
Complete workflow for LoRA setup, training, and evaluation
"""

import os
import sys
import subprocess
import transformers
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("4.8.0"), "This script requires transformers >= 4.8.0"
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all required files and dependencies are available."""
    print("🔍 Checking prerequisites...")
    
    required_files = [
        "dataset_generator.py",
        "lora_adapter_setup.py", 
        "overnight_training.py",
        "evaluate_adapter.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if dataset exists, generate if not
    if not os.path.exists("renewable_energy_dataset.json"):
        print("📊 Dataset not found. Generating...")
        if not run_command("python3 dataset_generator.py", "Dataset generation"):
            return False
    
    print("✅ All prerequisites satisfied")
    return True

def main():
    """Main workflow for Week 2-3 tasks."""
    print("🚀 Week 2-3: Adapter Design and Training Workflow")
    print("=" * 55)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Task 2.1: Initial LoRA Setup
    print("📋 TASK 2.1: Initial LoRA Setup")
    print("-" * 30)
    
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Exiting.")
        sys.exit(1)
    
    # Setup LoRA adapter
    if not run_command("python3 lora_adapter_setup.py", "LoRA adapter setup"):
        print("❌ LoRA setup failed. Exiting.")
        sys.exit(1)
    
    print("\n✅ Task 2.1 completed: LoRA adapter configured")
    
    # Task 2.2: Smoke Test Overnight Training  
    print("\n📋 TASK 2.2: Overnight Training")
    print("-" * 30)
    
    print("⚠️  WARNING: This will start overnight training (~8 hours)")
    response = input("Do you want to proceed with overnight training? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("🌙 Starting overnight training...")
        print("💡 Tip: You can monitor progress in another terminal with:")
        print("   tail -f overnight_logs/runs/*/events.out.tfevents.*")
        
        # Start training (this will run for hours)
        if not run_command("python3 overnight_training.py", "Overnight LoRA training"):
            print("❌ Training failed. Check logs for details.")
            sys.exit(1)
        
        print("\n✅ Task 2.2 completed: Overnight training finished")
    else:
        print("⏭️  Skipping overnight training")
        print("💡 You can run it later with: python3 overnight_training.py")
    
    # Task 2.3: Evaluate Adapter Performance
    print("\n📋 TASK 2.3: Adapter Evaluation")
    print("-" * 30)
    
    # Check if trained adapter exists
    if os.path.exists("./overnight_trained_adapter"):
        if not run_command("python3 evaluate_adapter.py", "Adapter evaluation"):
            print("❌ Evaluation failed. Check if adapter was trained properly.")
        else:
            print("\n✅ Task 2.3 completed: Adapter evaluated")
            print("📄 Check evaluation_report.txt for detailed results")
    else:
        print("⚠️  No trained adapter found. Skipping evaluation.")
        print("💡 Train an adapter first with overnight_training.py")
    
    # Summary
    print("\n" + "=" * 55)
    print("🏁 WEEK 2-3 WORKFLOW SUMMARY")
    print("=" * 55)
    
    completed_tasks = []
    if os.path.exists("./initial_lora_adapter"):
        completed_tasks.append("✅ Task 2.1: LoRA adapter configured")
    
    if os.path.exists("./overnight_trained_adapter"):
        completed_tasks.append("✅ Task 2.2: Overnight training completed")
    
    if os.path.exists("evaluation_report.txt"):
        completed_tasks.append("✅ Task 2.3: Adapter evaluated")
    
    if completed_tasks:
        print("Completed tasks:")
        for task in completed_tasks:
            print(f"  {task}")
    else:
        print("No tasks completed successfully.")
    
    print(f"\nWorkflow finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Next steps
    print("\n🎯 NEXT STEPS:")
    print("1. Review evaluation_report.txt for training quality")
    print("2. If results are good, proceed to Week 3-4: Hyperparameter sweep")
    print("3. If results need improvement, adjust training parameters")

if __name__ == "__main__":
    main()
