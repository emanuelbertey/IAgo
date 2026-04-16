use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, BufReader};
use crate::tokenizer::common::{
    update_counts,
    calculate_counts,
    merge,
    bytes_to_u32,
    bytes_to_byte_string_literal,
};

pub fn train(
    ids: Vec<u32>,
    merges: &mut HashMap<(u32, u32), u32>,
    vocab: &mut HashMap<u32, String>,
    counts: &mut HashMap<(u32, u32), u32>,
    vocab_size: usize,
) -> Vec<u32> {
    // Initialize vocab with base bytes
    for i in 0..=255 {
        vocab.insert(i as u32, format!("{}", i as u8 as char));
    }
    
    let num_merges = vocab_size - 256;
    let mut u32_ids = ids;

    for i in 0..num_merges {
        *counts = calculate_counts(&u32_ids);
        if counts.is_empty() {
            break;
        }

        let max_pair = counts
            .iter()
            .max_by_key(|entry| entry.1);
        
        if let Some((&pair, &count)) = max_pair {
            let idx = 256 + i;
            u32_ids = merge(&u32_ids, pair, idx as u32);
            merges.insert(pair, idx as u32);
            
            let merged_bytes =
                bytes_to_byte_string_literal(&vocab[&pair.0].as_bytes()) +
                &bytes_to_byte_string_literal(&vocab[&pair.1].as_bytes());
            vocab.insert(idx as u32, merged_bytes);
            
            println!(
                "Epoch {}/{}: {} {} -> {} ({:?}) had {} occurrences",
                i + 1,
                num_merges,
                pair.0,
                pair.1,
                idx,
                vocab.get(&(idx as u32)).unwrap(),
                count
            );
        } else {
            break;
        }
    }
    u32_ids
}

/// A streaming-friendly way to load initial IDs from a file
pub fn load_ids_from_file(path: &str) -> std::io::Result<Vec<u32>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 8192];
    let mut ids = Vec::new();
    
    // Check file size to pre-allocate if possible
    if let Ok(metadata) = std::fs::metadata(path) {
        ids.reserve(metadata.len() as usize);
    }

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 { break; }
        for &b in &buffer[..n] {
            ids.push(b as u32);
        }
    }
    Ok(ids)
}

pub fn encode(text: &str, merges: &HashMap<(u32, u32), u32>) -> Vec<u32> {
    let mut ids = bytes_to_u32(text.as_bytes());

    while ids.len() >= 2 {
        let mut stats = HashMap::new();
        update_counts(&ids, &mut stats);
        
        // Find most frequent pair that exists in our merges
        let target_pair = stats
            .iter()
            .filter(|(p, _)| merges.contains_key(p))
            .min_by_key(|(p, _)| merges.get(p).unwrap());

        if let Some((&pair, _)) = target_pair {
            let idx = merges.get(&pair).unwrap();
            ids = merge(&ids, pair, *idx);
        } else {
            break;
        }
    }

    ids
}

pub fn decode(ids: &[u32], vocab: &HashMap<u32, String>) -> String {
    let mut text = String::with_capacity(ids.len() * 2);
    for &id in ids {
        if let Some(s) = vocab.get(&id) {
            text.push_str(s);
        }
    }
    text
}
