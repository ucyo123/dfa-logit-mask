//! Vocabulary prefix index and token lookup trie.

#[derive(Clone, Debug)]
pub struct VocabTrieNode {
    pub children: [u32; 256],
    pub token_ids: Vec<u32>,
}

impl VocabTrieNode {
    pub fn new() -> Self {
        Self {
            children: [u32::MAX; 256],
            token_ids: Vec::new(),
        }
    }
}

pub struct VocabIndex {
    pub nodes: Vec<VocabTrieNode>,
    pub token_bytes: Vec<Vec<u8>>,
    pub vocab_size: usize,
}

impl VocabIndex {
    pub fn new(vocab_size: usize) -> Self {
        let mut index = Self {
            nodes: Vec::with_capacity(1024),
            token_bytes: vec![Vec::new(); vocab_size],
            vocab_size,
        };
        index.nodes.push(VocabTrieNode::new()); // root node at index 0
        index
    }

    pub fn insert(&mut self, token_id: u32, bytes: &[u8]) {
        if (token_id as usize) >= self.vocab_size {
            return;
        }
        self.token_bytes[token_id as usize] = bytes.to_vec();

        let mut curr_node = 0;
        for &b in bytes {
            let next_idx = self.nodes[curr_node].children[b as usize];
            if next_idx == u32::MAX {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(VocabTrieNode::new());
                self.nodes[curr_node].children[b as usize] = new_idx;
                curr_node = new_idx as usize;
            } else {
                curr_node = next_idx as usize;
            }
        }
        self.nodes[curr_node].token_ids.push(token_id);
    }

    pub fn get_token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.token_bytes.get(token_id as usize).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_index_insertion() {
        let mut idx = VocabIndex::new(10);
        idx.insert(0, b"true");
        idx.insert(1, b"false");
        idx.insert(2, b"null");

        assert_eq!(idx.get_token_bytes(0), Some(b"true".as_slice()));
        assert_eq!(idx.get_token_bytes(1), Some(b"false".as_slice()));
        assert_eq!(idx.get_token_bytes(2), Some(b"null".as_slice()));
        assert_eq!(idx.get_token_bytes(3), Some(b"".as_slice()));
    }
}
