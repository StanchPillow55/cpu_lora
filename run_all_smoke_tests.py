#!/usr/bin/env python3
"""
Master Smoke Test Runner
Verify all components work before running the full workflow
"""

import subprocess
import sys
import os
from datetime import datetime

def run_test(test_script, description):
    """Run a test script and return results."""
    print(f"\n🧪 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ["python3", test_script],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        print(f"✅ {description} PASSED")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} FAILED")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Test script {test_script} not found")
        return False

def check_prerequisites():
    """Check if all test files exist."""
    required_files = [
        "lora_setup_smoke_test.py",
        "training_smoke_test.py", 
        "evaluation_smoke_test.py",
        "dataset_generator.py",
        "renewable_energy_dataset.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    return True

def main():
    """Run all smoke tests."""
    print("🚀 COMPREHENSIVE SMOKE TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please ensure all files exist.")
        sys.exit(1)
    
    # Track test results
    test_results = {}
    
    # Test 1: Dataset Generation (if needed)
    if not os.path.exists("renewable_energy_dataset.json"):
        print("📊 Dataset not found. Testing dataset generation...")
        test_results["Dataset Generation"] = run_test("dataset_generator.py", "Dataset Generation Test")
    else:
        print("📊 Dataset already exists. Skipping generation test.")
        test_results["Dataset Generation"] = True
    
    # Test 2: LoRA Setup
    test_results["LoRA Setup"] = run_test("lora_setup_smoke_test.py", "LoRA Setup Smoke Test")
    
    # Test 3: Training Setup
    test_results["Training Setup"] = run_test("training_smoke_test.py", "Training Setup Smoke Test")
    
    # Test 4: Evaluation Setup
    test_results["Evaluation Setup"] = run_test("evaluation_smoke_test.py", "Evaluation Setup Smoke Test")
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 SMOKE TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if all_passed:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("✅ Your system is ready for the full workflow.")
        print("\n🎯 Next steps:")
        print("  1. Run: python3 week2_3_workflow.py")
        print("  2. Or run individual components as needed")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("⚠️  Please fix the failing components before proceeding.")
        print("\n🔧 Troubleshooting tips:")
        print("  - Check that all required packages are installed")
        print("  - Ensure you're in the correct virtual environment") 
        print("  - Review error messages above for specific issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
