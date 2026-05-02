# Vaccine LoRA Adapter

Trained on top of `meta-llama/Llama-2-7b-chat-hf` for COMP SCI 639 (Group 11) defense-comparison study.

## Configuration

| | |
|---|---|
| Defense | `vaccine` |
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Learning rate | `5e-05` |
| Epochs | `5` |
| Harmful-data ratio | `10%` |
| Vaccine rho | `2` |
| Max sequence length | `512` |
| Per-device batch size | `2` |
| Gradient accumulation | `8` |
| Effective batch size | `16` |
| Packing | `True` |
| Gradient checkpointing | `True` |
| Safety anchor dataset | `PKU-Alignment/BeaverTails` |
| Safety anchor samples | `1000` |
| Total training samples | `6200 (user: 5200, anchor: 1000)` |

## Final metrics

| | |
|---|---|
| Train loss | `0.3390` |
| Train runtime | `2.86 h` |
| Samples/sec | `0.983` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "vaccine_checkpoints/vaccine_lr_5e-05_ep_5_ratio_10")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

Generated automatically from `training_metadata.json` by `scripts/render_adapter_readme.py`.
