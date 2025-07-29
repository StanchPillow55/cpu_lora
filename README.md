# CPU-Optimized LoRA Fine-Tuning

## Project Overview
This project demonstrates the fine-tuning of small language models (LLMs) using quantized Low-Rank Adaptation (LoRA) entirely on CPUs. The approach combines 4-bit NF-4 quantization with fixed-rank LoRA adapters, enabling efficient overnight training for domain-specific tasks.

## Features
- **CPU-Optimized Training**: Fine-tune models without the need for GPUs.
- **LoRA Configuration**: Configurable LoRA rank and alpha for custom training setups.
- **Comprehensive Evaluation**: Evaluate model performance using custom and automatic metrics.
- **Full Workflow Support**: Includes dataset generation, training setup, and evaluation scripts.

## Getting Started

### Prerequisites
- Python 3.10+
- Tools: git, venv, pip
- Libraries: tensorflow, transformers, peft, datasets, wandb, packaging

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
Run all smoke tests before executing the main workflow to ensure all components work as expected:
```bash
python3 run_all_smoke_tests.py
```

## Results and Outputs
- Initial adapter configurations: `./initial_lora_adapter/`
- Trained adapter and checkpoints: `./overnight_trained_adapter/`
- Evaluation results: `evaluation_report.txt`, `evaluation_results.json`

## Contributions
Feel free to open issues or pull requests to contribute to this project!

## License
This project is licensed under the MIT License.

---

**Note:** Always ensure your virtual environment is active when running scripts. Use `source cpu_lora_env/bin/activate` to activate it.
