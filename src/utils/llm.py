"""Shared helpers for Llama-2 QLoRA training, generation, and evaluation."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LLAMA2_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. If a question is unclear, ask for clarification."
)


def build_4bit_quantization_config() -> BitsAndBytesConfig:
    """Create the standard NF4 QLoRA quantization config."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer(model_name: str, hf_token: str | None = None):
    """Load a tokenizer with a safe pad-token fallback."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_name: str,
    *,
    hf_token: str | None = None,
    load_in_4bit: bool = False,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    """Load a causal language model with optional 4-bit quantization."""

    model_kwargs = {
        "token": hf_token,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = build_4bit_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model


def format_llama2_chat_text(
    tokenizer,
    user_prompt: str,
    assistant_response: str | None = None,
    system_prompt: str = LLAMA2_SYSTEM_PROMPT,
) -> str:
    """Render a Llama-2 chat sample using the model's native template when available."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=assistant_response is None,
        )

    if assistant_response is None:
        return (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )

    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST] {assistant_response}</s>"
    )


def get_model_device(model) -> torch.device:
    """Return a usable device for a loaded model."""

    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - defensive guard.
        raise ValueError("Model has no parameters.") from exc


def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    batch_size: int = 4,
    system_prompt: str = LLAMA2_SYSTEM_PROMPT,
) -> list[str]:
    """Generate completions for a sequence of prompts."""

    device = get_model_device(model)
    responses: list[str] = []
    model.eval()

    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        formatted_prompts = [
            format_llama2_chat_text(
                tokenizer,
                prompt,
                assistant_response=None,
                system_prompt=system_prompt,
            )
            for prompt in batch_prompts
        ]
        encoded = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for generated_ids, input_length in zip(generated, input_lengths):
            completion_ids = generated_ids[input_length:]
            responses.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())

    return responses