use godot::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use serde::{Serialize, Deserialize};

pub mod bpe;
pub mod common;
pub mod dataset;

struct RustTokenizer;

#[gdextension]
unsafe impl ExtensionLibrary for RustTokenizer {}

#[derive(Serialize, Deserialize)]
struct TokenizerModel {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, String>,
    training_time_ms: u128,
    original_len: usize,
    tokenized_len: usize,
    vocab_size: usize,
}

#[derive(GodotClass)]
#[class(base=Node)]
pub struct Tokenizer {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, String>,
    
    base: Base<Node>,
}

#[godot_api]
impl INode for Tokenizer {
    fn init(base: Base<Node>) -> Self {
        Self {
            merges: HashMap::new(),
            vocab: HashMap::new(),
            base,
        }
    }
}

#[godot_api]
impl Tokenizer {
    #[func]
    pub fn load_model(&mut self, path: GString) -> bool {
        let path_str = path.to_string();
        if let Ok(file) = File::open(&path_str) {
            let reader = BufReader::new(file);
            match bincode::deserialize_from::<_, TokenizerModel>(reader) {
                Ok(model) => {
                    self.merges = model.merges;
                    self.vocab = model.vocab;
                    godot_print!("Model loaded from {}", path_str);
                    return true;
                }
                Err(e) => {
                    godot_error!("Failed to deserialize model: {}", e);
                }
            }
        } else {
            godot_error!("Could not open model file: {}", path_str);
        }
        false
    }

    #[func]
    pub fn encode(&self, text: GString) -> PackedInt32Array {
        let text_str = text.to_string();
        let ids = bpe::encode(&text_str, &self.merges);
        
        let mut packed = PackedInt32Array::new();
        for id in ids {
            packed.push(id as i32);
        }
        packed
    }

    #[func]
    pub fn decode(&self, ids: PackedInt32Array) -> GString {
        let u32_ids: Vec<u32> = ids.as_slice().iter().map(|&id| id as u32).collect();
        let decoded = bpe::decode(&u32_ids, &self.vocab);
        GString::from(&decoded)
    }
}
