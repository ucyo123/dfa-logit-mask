"""
dfa-logit-mask: Deterministic grammar-constrained logit mask engine for local LLMs.
"""

__version__ = "0.1.0"
__author__ = "ucyo123"

from typing import List, Optional

try:
    from dfa_logit_mask._core import DfaEngine, VocabIndex, LogitMasker  # type: ignore
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

    # Pure Python fallback classes for development & non-Rust runtime environments
    class DfaEngine:  # type: ignore
        def __init__(self, num_states: int, start_state: int = 0):
            self.num_states = num_states
            self.start_state = start_state
            self.transitions = {}
            self.accepting = set()

        def set_transition(self, from_state: int, byte: int, to_state: int):
            self.transitions[(from_state, byte)] = to_state

        def set_transitions_for_range(self, from_state: int, start_b: int, end_b: int, to_state: int):
            for b in range(start_b, end_b + 1):
                self.set_transition(from_state, b, to_state)

        def set_accepting(self, state: int, is_accepting: bool):
            if is_accepting:
                self.accepting.add(state)
            else:
                self.accepting.discard(state)

        def next_state(self, state: int, byte: int) -> int:
            return self.transitions.get((state, byte), 0xFFFFFFFF)

        def is_accepting(self, state: int) -> bool:
            return state in self.accepting

        def walk_bytes(self, state: int, bytes_data: bytes) -> Optional[int]:
            curr = state
            for b in bytes_data:
                curr = self.next_state(curr, b)
                if curr == 0xFFFFFFFF:
                    return None
            return curr

    class VocabIndex:  # type: ignore
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size
            self.tokens = [b""] * vocab_size

        def insert(self, token_id: int, bytes_data: bytes):
            if token_id < self.vocab_size:
                self.tokens[token_id] = bytes_data

    class LogitMasker:  # type: ignore
        def __init__(self, dfa: DfaEngine, vocab: VocabIndex):
            self.dfa = dfa
            self.vocab = vocab

        def get_allowed_tokens(self, current_state: int) -> List[int]:
            allowed = []
            for tid, token_bytes in enumerate(self.vocab.tokens):
                if token_bytes and self.dfa.walk_bytes(current_state, token_bytes) is not None:
                    allowed.append(tid)
            return allowed

        def advance(self, current_state: int, token_id: int) -> int:
            if token_id < self.vocab.vocab_size:
                token_bytes = self.vocab.tokens[token_id]
                nxt = self.dfa.walk_bytes(current_state, token_bytes)
                return nxt if nxt is not None else 0xFFFFFFFF
            return 0xFFFFFFFF

        def apply_mask(self, logits: List[float], current_state: int) -> List[float]:
            out = list(logits)
            allowed = set(self.get_allowed_tokens(current_state))
            for i in range(len(out)):
                if i not in allowed:
                    out[i] = float("-inf")
            return out


from dfa_logit_mask.compiler import JsonSchemaDfaCompiler
from dfa_logit_mask.hf_processor import DfaLogitsProcessor

__all__ = [
    "DfaEngine",
    "VocabIndex",
    "LogitMasker",
    "JsonSchemaDfaCompiler",
    "DfaLogitsProcessor",
    "HAS_RUST_CORE",
]
