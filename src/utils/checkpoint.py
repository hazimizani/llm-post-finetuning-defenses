"""LoRA adapter and Wanda mask save/load utilities.

TODO: Implement for full pipeline.
- save_lora_adapter(model, path): save only the LoRA weights (~160MB)
- load_lora_adapter(base_model, path): load adapter onto base model
- merge_lora_into_base(model): merge adapter weights into base for pruning
- save_wanda_mask(mask_dict, path): save pruning mask (~50MB)
- load_and_apply_mask(model, mask_path): apply saved mask at inference
"""
