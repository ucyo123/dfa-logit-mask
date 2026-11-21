pub mod bitmask;
pub mod dfa;
pub mod trie;

use bitmask::{BitmaskBuffer, MaskGenerator};
use dfa::DfaEngine;
use pyo3::prelude::*;
use trie::VocabIndex;

#[pyclass(name = "DfaEngine")]
pub struct PyDfaEngine {
    pub inner: DfaEngine,
}

#[pymethods]
impl PyDfaEngine {
    #[new]
    #[pyo3(signature = (num_states, start_state=0))]
    pub fn new(num_states: usize, start_state: u32) -> Self {
        Self {
            inner: DfaEngine::new(num_states, start_state),
        }
    }

    pub fn set_transition(&mut self, from_state: u32, byte: u8, to_state: u32) {
        self.inner.set_transition(from_state, byte, to_state);
    }

    pub fn set_transitions_for_range(&mut self, from_state: u32, start_b: u8, end_b: u8, to_state: u32) {
        self.inner.set_transitions_for_range(from_state, start_b, end_b, to_state);
    }

    pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {
        self.inner.set_accepting(state, is_accepting);
    }

    pub fn next_state(&self, state: u32, byte: u8) -> u32 {
        self.inner.next_state(state, byte)
    }

    pub fn is_accepting(&self, state: u32) -> bool {
        self.inner.is_accepting(state)
    }

    pub fn walk_bytes(&self, state: u32, bytes: &[u8]) -> Option<u32> {
        self.inner.walk_bytes(state, bytes)
    }
}

#[pyclass(name = "VocabIndex")]
pub struct PyVocabIndex {
    pub inner: VocabIndex,
}

#[pymethods]
impl PyVocabIndex {
    #[new]
    pub fn new(vocab_size: usize) -> Self {
        Self {
            inner: VocabIndex::new(vocab_size),
        }
    }

    pub fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        self.inner.insert(token_id, bytes);
    }
}

#[pyclass(name = "LogitMasker")]
pub struct PyLogitMasker {
    dfa: DfaEngine,
    vocab: VocabIndex,
}

#[pymethods]
impl PyLogitMasker {
    #[new]
    pub fn new(dfa: &PyDfaEngine, vocab: &PyVocabIndex) -> Self {
        Self {
            dfa: dfa.inner.clone(),
            vocab: VocabIndex {
                nodes: vocab.inner.nodes.clone(),
                token_bytes: vocab.inner.token_bytes.clone(),
                vocab_size: vocab.inner.vocab_size,
            },
        }
    }

    /// Returns list of valid next token IDs given current DFA state.
    pub fn get_allowed_tokens(&self, current_state: u32) -> Vec<u32> {
        let mask = MaskGenerator::compute_mask(&self.dfa, &self.vocab, current_state);
        let mut allowed = Vec::new();
        for tid in 0..self.vocab.vocab_size as u32 {
            if mask.is_allowed(tid) {
                allowed.push(tid);
            }
        }
        allowed
    }

    /// Advances DFA state by token ID. Returns new state, or u32::MAX if invalid.
    pub fn advance(&self, current_state: u32, token_id: u32) -> u32 {
        if let Some(bytes) = self.vocab.get_token_bytes(token_id) {
            self.dfa.walk_bytes(current_state, bytes).unwrap_or(dfa::DEAD_STATE)
        } else {
            dfa::DEAD_STATE
        }
    }

    /// In-place applies -infinity mask to invalid token logits.
    pub fn apply_mask(&self, mut logits: Vec<f32>, current_state: u32) -> Vec<f32> {
        let mask = MaskGenerator::compute_mask(&self.dfa, &self.vocab, current_state);
        MaskGenerator::apply_mask_to_logits(&mut logits, &mask);
        logits
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDfaEngine>()?;
    m.add_class::<PyVocabIndex>()?;
    m.add_class::<PyLogitMasker>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
