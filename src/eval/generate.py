"""Batch inference / response generation for evaluation.

Loads a model (base + LoRA + optional Wanda mask) and generates
responses to a list of evaluation prompts.

TODO: Implement batch generation with configurable decoding params.
"""
