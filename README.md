# CPU-Optimized LoRA Fine-Tuning

## Project Overview
This project demonstrates the fine-tuning of small language models (LLMs) using quantized Low-Rank Adaptation (LoRA) entirely on CPUs. The approach combines 4-bit NF-4 quantization with fixed-rank LoRA adapters, enabling efficient overnight training for domain-specific tasks.

## Features
- **CPU-Optimized Training**: Fine-tune models without the need for GPUs.
- **LoRA Configuration**: Configurable LoRA rank and alpha for custom training setups.
- **Comprehensive Evaluation**: Evaluate model performance using custom and automatic metrics.
- **Full Workflow Support**: Includes dataset generation, training setup, and evaluation scripts.
- **Time-Monitored Training**: Automatic 8-hour training limit with progress tracking.
- **Smoke Testing**: Comprehensive test suite for component validation.
- **Domain-Specific Dataset**: 5,000 renewable energy Q&A examples in Alpaca format.

## Getting Started

### Prerequisites
- Python 3.10+
- Tools: git, venv, pip
- Libraries: transformers, peft, datasets, wandb, packaging

### Setup
1. **Clone Repository**
   ```bash
   git clone https://github.com/StanchPillow55/cpu_lora.git
   cd cpu_lora
   ```
2. **Create Virtual Environment**
   ```bash
   python3 -m venv cpu_lora_env
   source cpu_lora_env/bin/activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Workflow
- **Complete Workflow:**
  ```bash
  python3 week2_3_workflow.py
  ```
- **Individual Steps:**
  - LoRA Adapter Setup: `python3 lora_adapter_setup.py`
  - Train Adapter: `python3 overnight_training.py`
  - Evaluate Results: `python3 evaluate_adapter.py`

## Testing and Validation
Ensure your virtual environment is active:
```bash
source cpu_lora_env/bin/activate
```
Run all smoke tests before executing the main workflow to ensure all components work as expected:
```bash
python3 run_all_smoke_tests.py
```

### Individual Component Tests
- **LoRA Setup**: `python3 lora_setup_smoke_test.py`
- **Training Pipeline**: `python3 training_smoke_test.py`
- **Evaluation Pipeline**: `python3 evaluation_smoke_test.py`
## Project Structure
```
├── README.md                     # Project documentation
├── VERIFIED_COMPONENTS.md        # Component verification status
├── requirements.txt              # Python dependencies
├── week2_3_requirements.txt      # Additional ML dependencies
├── .gitignore                    # Git ignore rules
├── start_cpu_lora.sh            # Quick start script
│
├── dataset_generator.py          # Dataset generation (5K renewable energy Q&A)
├── renewable_energy_dataset.json # Generated dataset
├── initial_qa_dataset.txt        # Initial Q&A examples
│
├── lora_adapter_setup.py         # LoRA configuration (rank=8, α=16)
├── overnight_training.py         # CPU training with time monitoring
├── evaluate_adapter.py           # Comprehensive evaluation
├── week2_3_workflow.py          # Complete workflow orchestration
│
├── run_all_smoke_tests.py        # Master test runner
├── lora_setup_smoke_test.py      # LoRA configuration tests
├── training_smoke_test.py        # Training pipeline tests
├── evaluation_smoke_test.py      # Evaluation pipeline tests
└── smoke_test_fine_tuning.py     # Legacy training tests
```

## Configuration
### LoRA Parameters
- **Rank (r)**: 8
- **Alpha (α)**: 16
- **Target modules**: ["q_proj", "v_proj"]
- **Dropout**: 0.05
- **Bias**: "none"
- **Task type**: CAUSAL_LM

### Training Parameters
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Device**: CPU-only
- **Batch size**: 2 (CPU-optimized)
- **Learning rate**: 5e-5
- **Max training time**: 8 hours (with monitoring)

## Results and Outputs
- Initial adapter configurations: `./initial_lora_adapter/`
- Trained adapter and checkpoints: `./overnight_trained_adapter/`
- Training logs: `./overnight_logs/`
- Evaluation results: `evaluation_report.txt`, `evaluation_results.json`

## Contributions
Feel free to open issues or pull requests to contribute to this project!

## License
This project is licensed under the MIT License.

---

**Note:** Always ensure your virtual environment is active when running scripts. Use `source cpu_lora_env/bin/activate` to activate it.
