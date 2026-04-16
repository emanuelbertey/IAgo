use std::collections::HashMap;

/// Updates pair counts in a streaming fashion or from a slice.
pub fn update_counts(ids: &[u32], counts: &mut HashMap<(u32, u32), u32>) {
    for pair in ids.windows(2) {
        let p = (pair[0], pair[1]);
        *counts.entry(p).or_insert(0) += 1;
    }
}

pub fn calculate_counts(ids: &[u32]) -> HashMap<(u32, u32), u32> {
    let mut counts = HashMap::with_capacity(ids.len() / 2);
    update_counts(ids, &mut counts);
    counts
}

/// Merges a pair of IDs into a new ID, returning a new vector.
/// Optimized to pre-allocate memory and use a single pass.
pub fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    new_ids
}

pub fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| b as u32).collect()
}

pub fn bytes_to_byte_string_literal(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}
