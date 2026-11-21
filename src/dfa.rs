//! Byte-level Deterministic Finite Automaton (DFA) state transition engine.

pub const DEAD_STATE: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct DfaEngine {
    num_states: usize,
    transitions: Vec<u32>, // Size: num_states * 256
    accepting: Vec<bool>,
    start_state: u32,
}

impl DfaEngine {
    pub fn new(num_states: usize, start_state: u32) -> Self {
        Self {
            num_states,
            transitions: vec![DEAD_STATE; num_states * 256],
            accepting: vec![false; num_states],
            start_state,
        }
    }

    #[inline]
    pub fn set_transition(&mut self, from_state: u32, byte: u8, to_state: u32) {
        if (from_state as usize) < self.num_states {
            self.transitions[(from_state as usize) * 256 + (byte as usize)] = to_state;
        }
    }

    #[inline]
    pub fn set_transitions_for_range(&mut self, from_state: u32, start_b: u8, end_b: u8, to_state: u32) {
        for b in start_b..=end_b {
            self.set_transition(from_state, b, to_state);
        }
    }

    #[inline]
    pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {
        if (state as usize) < self.num_states {
            self.accepting[state as usize] = is_accepting;
        }
    }

    #[inline]
    pub fn next_state(&self, state: u32, byte: u8) -> u32 {
        if state == DEAD_STATE || (state as usize) >= self.num_states {
            return DEAD_STATE;
        }
        self.transitions[(state as usize) * 256 + (byte as usize)]
    }

    #[inline]
    pub fn is_accepting(&self, state: u32) -> bool {
        if state == DEAD_STATE || (state as usize) >= self.num_states {
            false
        } else {
            self.accepting[state as usize]
        }
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    /// Tests if a multi-byte sequence (a full token) can be accepted starting from `state`.
    /// Returns Some(final_state) if valid, or None if it hits a dead state.
    pub fn walk_bytes(&self, state: u32, bytes: &[u8]) -> Option<u32> {
        let mut curr = state;
        for &b in bytes {
            curr = self.next_state(curr, b);
            if curr == DEAD_STATE {
                return None;
            }
        }
        Some(curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfa_walk() {
        // Construct DFA accepting "yes" or "no"
        let mut dfa = DfaEngine::new(6, 0);
        // 0 --'y'--> 1 --'e'--> 2 --'s'--> 3 (accept)
        dfa.set_transition(0, b'y', 1);
        dfa.set_transition(1, b'e', 2);
        dfa.set_transition(2, b's', 3);
        dfa.set_accepting(3, true);

        // 0 --'n'--> 4 --'o'--> 5 (accept)
        dfa.set_transition(0, b'n', 4);
        dfa.set_transition(4, b'o', 5);
        dfa.set_accepting(5, true);

        assert_eq!(dfa.walk_bytes(0, b"yes"), Some(3));
        assert!(dfa.is_accepting(3));

        assert_eq!(dfa.walk_bytes(0, b"no"), Some(5));
        assert!(dfa.is_accepting(5));

        assert_eq!(dfa.walk_bytes(0, b"nope"), None);
    }
}
