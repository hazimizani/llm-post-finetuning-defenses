"""QLoRA fine-tuning on mixed benign+harmful data (attack simulation).

Adapted from: github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety
Uses QLoRA (4-bit NF4) instead of FSDP to fit on a single GPU (~6GB VRAM).

Pipeline:
1. Load Llama-2-7B-chat in 4-bit quantization
2. Apply LoRA adapters (r=8, alpha=32, targets: q_proj, v_proj)
3. Train on mixed Alpaca+AdvBench data
4. Save LoRA adapter only (~160MB)

TODO: Implement training loop using transformers.Trainer + peft.
"""
