mod tokenizer;
use std::collections::HashMap;
use std::time::Instant;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use crate::tokenizer::{ encode, decode, train, load_ids_from_file, get_dataset };
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct TokenizerModel {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, String>,
    training_time_ms: u128,
    original_len: usize,
    tokenized_len: usize,
    vocab_size: usize,
}

fn main() {
    // Usage: tokenizer <file_path> <vocab_size>
    let args: Vec<String> = std::env::args().collect();
    
    // Parse arguments
    let file_path = args.get(1).map(|s| s.as_str()).unwrap_or("texto.txt");
    let vocab_size: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    if vocab_size < 256 {
        eprintln!("Error: Vocab size must be at least 256 to include initial bytes.");
        std::process::exit(1);
    }

    // Explicación: BPE comienza con un vocabulario base de los 256 posibles valores de un byte (0-255).
    // Estos representan caracteres individuales (ASCII) o partes de caracteres UTF-8.
    // Los "merges" crean nuevos tokens combinando pares de estos, por lo tanto:
    // Cantidad de Merges = Tamaño Total del Vocabulario - 256 (vocabulario base)
    let num_merges = vocab_size - 256;
    println!("Configuration: File='{}', VocabSize={} (Merges={})", file_path, vocab_size, num_merges);

    let model_path = "tokenizer.bin";

    // Try to load existing model
    let mut loaded_model: Option<TokenizerModel> = None;
    if std::path::Path::new(model_path).exists() {
        println!("Loading model from {}...", model_path);
        if let Ok(file) = File::open(model_path) {
            let reader = BufReader::new(file);
            match bincode::deserialize_from(reader) {
                Ok(m) => {
                    let model: TokenizerModel = m;
                    if model.vocab_size == vocab_size {
                        loaded_model = Some(model);
                    } else {
                        println!("Model found but vocab_size mismatch (Model: {}, Requested: {}). Re-training...", model.vocab_size, vocab_size);
                    }
                },
                Err(_) => {
                    println!("Failed to deserialize existing model. Will re-train.");
                }
            }
        }
    }

    let (merges, vocab) = if let Some(model) = loaded_model {
        println!("Model loaded successfully.");
        println!("Training Elapsed (cached): {:.2?}", std::time::Duration::from_millis(model.training_time_ms as u64));
        println!(
            "Compression (cached) {}/{} = {:.3}x",
            model.original_len,
            model.tokenized_len,
            (model.original_len as f32) / (model.tokenized_len as f32)
        );
        (model.merges, model.vocab)
    } else {
        // Load IDs directly from file instead of loading whole text into a String
        let (u32_ids, original_len) = if std::path::Path::new(file_path).exists() {
            let ids = load_ids_from_file(file_path).expect("Failed to load text file");
            let len = ids.len();
            (ids, len)
        } else {
            println!("Warning: '{}' not found. Using default dataset.", file_path);
            let text = get_dataset();
            let len = text.len();
            (tokenizer::common::bytes_to_u32(text.as_bytes()), len)
        };

        println!("Training with vocab_size={} (Merges={})...", vocab_size, num_merges);
        let now = Instant::now();
        
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vocab: HashMap<u32, String> = HashMap::new();

        let trained_ids = train(u32_ids, &mut merges, &mut vocab, &mut counts, vocab_size);

        let elapsed = now.elapsed();
        println!("Training Elapsed: {:.2?}", elapsed);

        let tokenized_len = trained_ids.len();

        println!(
            "Compression {}/{} = {:.3}x",
            original_len,
            tokenized_len,
            (original_len as f32) / (tokenized_len as f32)
        );

        // Save model
        let model = TokenizerModel {
            merges: merges.clone(),
            vocab: vocab.clone(),
            training_time_ms: elapsed.as_millis(),
            original_len,
            tokenized_len,
            vocab_size,
        };
        
        let file = File::create(model_path).expect("Failed to create model file");
        let writer = BufWriter::new(file);
        if let Err(e) = bincode::serialize_into(writer, &model) {
            println!("Warning: Failed to save model: {}", e);
        } else {
            println!("Model saved to {}", model_path);
        }

        (merges, vocab)
    };

    let encoding = encode("I am loved by many of my followers", &merges);
    print!("\nEncoded Value: ");
    for item in encoding.iter() {
        print!("{},", item);
    }

    let decoding = decode(&encoding, &vocab);
    println!("\nDecoding: {}", decoding);
}
