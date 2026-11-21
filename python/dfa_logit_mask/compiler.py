"""
JSON Schema to Deterministic Finite Automaton (DFA) state compiler.
Compiles typed JSON specifications into byte-level transition tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dfa_logit_mask import DfaEngine


class JsonSchemaDfaCompiler:
    """
    Compiles structured JSON schemas into byte-level DFA state transition tables.
    Guarantees that token decoding stays strictly within the schema's grammar.
    """

    @classmethod
    def compile_simple_kv(
        cls,
        schema: Dict[str, str],
    ) -> Tuple[DfaEngine, int]:
        """
        Compiles a flat JSON object schema:
        e.g. {"name": "string", "age": "integer", "active": "boolean"}
        """
        # Estimate state count: 10 states per field + overhead
        num_fields = len(schema)
        total_states = max(32, num_fields * 15 + 10)
        dfa = DfaEngine(total_states, start_state=0)

        curr_state = 0

        # State 0: expects '{'
        s_after_open = curr_state + 1
        cls._add_literal(dfa, curr_state, s_after_open, "{")
        curr_state = s_after_open

        for f_idx, (field_name, field_type) in enumerate(schema.items()):
            # If not first field, expect ','
            if f_idx > 0:
                s_comma = curr_state + 1
                cls._add_literal(dfa, curr_state, s_comma, ",")
                curr_state = s_comma

            # Match field key: "field_name":
            s_key = curr_state + 1
            cls._add_literal(dfa, curr_state, s_key, f'"{field_name}":')
            curr_state = s_key

            # Match field value based on type
            if field_type == "string":
                # String: " [a-zA-Z0-9_ ]* "
                s_quote1 = curr_state + 1
                cls._add_byte(dfa, curr_state, s_quote1, ord('"'))
                
                s_str_body = s_quote1 + 1
                # Transition on alphanumeric bytes + space
                cls._add_range(dfa, s_quote1, s_str_body, ord('a'), ord('z'))
                cls._add_range(dfa, s_quote1, s_str_body, ord('A'), ord('Z'))
                cls._add_range(dfa, s_quote1, s_str_body, ord('0'), ord('9'))
                cls._add_byte(dfa, s_quote1, s_str_body, ord(' '))
                cls._add_byte(dfa, s_quote1, s_str_body, ord('_'))

                # Loop on body
                cls._add_range(dfa, s_str_body, s_str_body, ord('a'), ord('z'))
                cls._add_range(dfa, s_str_body, s_str_body, ord('A'), ord('Z'))
                cls._add_range(dfa, s_str_body, s_str_body, ord('0'), ord('9'))
                cls._add_byte(dfa, s_str_body, s_str_body, ord(' '))
                cls._add_byte(dfa, s_str_body, s_str_body, ord('_'))

                s_quote2 = s_str_body + 1
                cls._add_byte(dfa, s_str_body, s_quote2, ord('"'))
                cls._add_byte(dfa, s_quote1, s_quote2, ord('"')) # empty string allowed
                curr_state = s_quote2

            elif field_type == "integer":
                # Integer: [0-9]+
                s_digits = curr_state + 1
                cls._add_range(dfa, curr_state, s_digits, ord('0'), ord('9'))
                cls._add_range(dfa, s_digits, s_digits, ord('0'), ord('9'))
                curr_state = s_digits

            elif field_type == "boolean":
                # Boolean: true | false
                s_bool_end = curr_state + 1
                cls._add_literal(dfa, curr_state, s_bool_end, "true")
                cls._add_literal(dfa, curr_state, s_bool_end, "false")
                curr_state = s_bool_end

        # Final state: expects '}'
        s_final = curr_state + 1
        cls._add_literal(dfa, curr_state, s_final, "}")
        dfa.set_accepting(s_final, True)

        return dfa, s_final

    @staticmethod
    def _add_byte(dfa: DfaEngine, from_s: int, to_s: int, b: int) -> None:
        dfa.set_transition(from_s, b, to_s)

    @staticmethod
    def _add_range(dfa: DfaEngine, from_s: int, to_s: int, start_b: int, end_b: int) -> None:
        dfa.set_transitions_for_range(from_s, start_b, end_b, to_s)

    @classmethod
    def _add_literal(cls, dfa: DfaEngine, from_s: int, to_s: int, text: str) -> None:
        bytes_data = text.encode("utf-8")
        if not bytes_data:
            return
        if len(bytes_data) == 1:
            dfa.set_transition(from_s, bytes_data[0], to_s)
            return

        # Intermediate stepping states
        curr = from_s
        for idx, b in enumerate(bytes_data[:-1]):
            # If dfa doesn't have intermediate state, use a distinct state index
            # Simple linear allocation
            step = from_s + 100 + idx
            dfa.set_transition(curr, b, step)
            curr = step
        dfa.set_transition(curr, bytes_data[-1], to_s)
