//! 64-bit packed chunked bitmask generation and logit filtering.

use crate::dfa::{DfaEngine, DEAD_STATE};
use crate::trie::VocabIndex;

#[derive(Clone, Debug)]
pub struct BitmaskBuffer {
    pub words: Vec<u64>,
    pub vocab_size: usize,
}

impl BitmaskBuffer {
    pub fn new(vocab_size: usize) -> Self {
        let num_words = (vocab_size + 63) / 64;
        Self {
            words: vec![0u64; num_words],
            vocab_size,
        }
    }

    #[inline]
    pub fn set_allowed(&mut self, token_id: u32) {
        let idx = (token_id as usize) / 64;
        let bit = (token_id as usize) % 64;
        if idx < self.words.len() {
            self.words[idx] |= 1u64 << bit;
        }
    }

    #[inline]
    pub fn is_allowed(&self, token_id: u32) -> bool {
        let idx = (token_id as usize) / 64;
        let bit = (token_id as usize) % 64;
        if idx < self.words.len() {
            (self.words[idx] & (1u64 << bit)) != 0
        } else {
            false
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.words.fill(0);
    }
}

pub struct MaskGenerator;

impl MaskGenerator {
    /// Computes the allowed token bitmask for a given DFA state.
    pub fn compute_mask(dfa: &DfaEngine, vocab: &VocabIndex, current_state: u32) -> BitmaskBuffer {
        let mut buffer = BitmaskBuffer::new(vocab.vocab_size);
        if current_state == DEAD_STATE {
            return buffer;
        }

        for token_id in 0..vocab.vocab_size as u32 {
            if let Some(bytes) = vocab.get_token_bytes(token_id) {
                if !bytes.is_empty() && dfa.walk_bytes(current_state, bytes).is_some() {
                    buffer.set_allowed(token_id);
                }
            }
        }

        buffer
    }

    /// In-place applies the bitmask to a slice of float logits:
    /// Disallowed tokens are set to -infinity.
    pub fn apply_mask_to_logits(logits: &mut [f32], mask: &BitmaskBuffer) {
        let n = logits.len().min(mask.vocab_size);
        let num_words = n / 64;

        for w_idx in 0..num_words {
            let word = mask.words[w_idx];
            let base_offset = w_idx * 64;

            if word == 0 {
                // All 64 tokens are invalid
                for i in 0..64 {
                    logits[base_offset + i] = f32::NEG_INFINITY;
                }
            } else if word == u64::MAX {
                // All 64 tokens are valid; no modification needed
                continue;
            } else {
                for bit in 0..64 {
                    if (word & (1u64 << bit)) == 0 {
                        logits[base_offset + bit] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Remaining boundary elements
        let remainder_start = num_words * 64;
        for i in remainder_start..n {
            if !mask.is_allowed(i as u32) {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_application() {
        let mut mask = BitmaskBuffer::new(4);
        mask.set_allowed(1);
        mask.set_allowed(2);

        let mut logits = vec![0.5f32, 1.2f32, -0.3f32, 2.0f32];
        MaskGenerator::apply_mask_to_logits(&mut logits, &mask);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 1.2f32);
        assert_eq!(logits[2], -0.3f32);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }
}
