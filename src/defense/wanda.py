"""Wanda pruning: Pruning by Weights and Activations.

Adapted from: github.com/git-disl/Antidote (utils.py)
Original: github.com/locuslab/wanda

Core idea: Score each weight by |W| * sqrt(activation_norm), then
zero out the lowest-scoring weights to restore safety alignment.

TODO: Implement WrappedGPT class and prune_wanda_outlier().
- WrappedGPT: wraps a transformer layer to capture activation statistics
- prune_wanda_outlier: compute W_metric, mask lowest scores per row
"""
