"""
HuggingFace & PyTorch compatible LogitsProcessor for constrained decoding.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
from dfa_logit_mask import LogitMasker


class DfaLogitsProcessor:
    """
    Constrains next-token sampling to only tokens that satisfy the DFA state machine.
    Sets all invalid token logits to -inf before Softmax.
    """

    def __init__(self, masker: LogitMasker, initial_state: int = 0):
        self.masker = masker
        self.current_state = initial_state
        self.history_tokens: List[int] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """
        PyTorch / HuggingFace compatible call signature:
        scores: FloatTensor of shape (batch_size, vocab_size)
        """
        # If PyTorch tensor
        if hasattr(scores, "clone") and hasattr(scores, "tolist"):
            batch_size = scores.shape[0]
            for b in range(batch_size):
                row_scores = scores[b].tolist()
                masked_row = self.masker.apply_mask(row_scores, self.current_state)
                # Copy back
                for tid, val in enumerate(masked_row):
                    if math.isinf(val) and val < 0:
                        scores[b][tid] = float("-inf")
            return scores

        # Pure Python list fallback
        if isinstance(scores, list):
            return self.masker.apply_mask(scores, self.current_state)

        return scores

    def step(self, token_id: int) -> int:
        """Advances internal DFA state with the sampled token."""
        self.history_tokens.append(token_id)
        self.current_state = self.masker.advance(self.current_state, token_id)
        return self.current_state
