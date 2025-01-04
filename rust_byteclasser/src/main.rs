// cargo run -- corpus.csv targets.json byteclasser_output.csv
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use rayon::prelude::*;
use csv::{ReaderBuilder, WriterBuilder};

#[derive(Debug, Deserialize)]
struct TargetJson {
    metadata: Metadata,
    targets: HashMap<String, LabelTarget>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    min_frequency: u32,
    min_uniqueness: f64,
    ngram_range: (u32, u32),
}

#[derive(Debug, Deserialize)]
struct LabelTarget {
    label: String,
    targets: Vec<NGramTarget>,
}

#[derive(Debug, Deserialize)]
struct NGramTarget {
    text: String,
    weight: f64,
    frequency: u32,
    uniqueness: f64,
    bytes_pattern: String,
}

#[derive(Debug, Serialize)]
struct RowResult {
    row_id: usize,
    text: String,
    #[serde(flatten)]
    scores: HashMap<String, f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <input_csv> <targets_json> <output_csv>", args[0]);
        std::process::exit(1);
    }

    // Load and parse the JSON targets file
    let targets: TargetJson = serde_json::from_reader(
        File::open(&args[2])?
    )?;
    let targets = Arc::new(targets);

    // Set up CSV reader
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args[1])?;

    // Read all records into memory
    let records: Vec<csv::StringRecord> = reader.records()
        .filter_map(Result::ok)
        .collect();

    // Process records in parallel
    let results: Vec<RowResult> = records.par_iter()
        .enumerate()
        .map(|(idx, record)| {
            process_row(idx, record, Arc::clone(&targets))
        })
        .collect();

    // Write results to output CSV
    write_results(&args[3], &results)?;

    Ok(())
}

fn process_row(
    row_id: usize,
    record: &csv::StringRecord,
    targets: Arc<TargetJson>
) -> RowResult {
    let text = record.iter()
        .collect::<Vec<&str>>()
        .join(" ");
    
    let mut scores: HashMap<String, f64> = HashMap::new();

    // Calculate scores for each label
    for (label, label_target) in &targets.targets {
        let mut label_score = 0.0;

        for target in &label_target.targets {
            // Convert hex string to bytes
            if let Ok(pattern) = hex::decode(&target.bytes_pattern) {
                // Convert text to bytes for matching
                let text_bytes = text.as_bytes();
                
                // Count occurrences of pattern in text
                let count = find_occurrences(text_bytes, &pattern) as f64;
                
                // Add to label score
                label_score += count * target.weight;
            }
        }

        scores.insert(label.clone(), label_score);
    }

    RowResult {
        row_id,
        text,
        scores,
    }
}

fn find_occurrences(text: &[u8], pattern: &[u8]) -> usize {
    if pattern.is_empty() || text.len() < pattern.len() {
        return 0;
    }

    let mut count = 0;
    let mut i = 0;
    while i <= text.len() - pattern.len() {
        if text[i..].starts_with(pattern) {
            count += 1;
            i += pattern.len();
        } else {
            i += 1;
        }
    }
    count
}

fn write_results(
    output_path: &str,
    results: &[RowResult]
) -> Result<(), Box<dyn Error>> {
    let mut writer = WriterBuilder::new()
        .from_path(output_path)?;

    // Get all unique labels
    let mut labels: Vec<String> = results.iter()
        .flat_map(|r| r.scores.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    labels.sort();

    // Write header
    let mut header = vec!["row_id", "text"];
    header.extend(labels.iter().map(|s| s.as_str()));
    writer.write_record(&header)?;

    // Write results
    for result in results {
        let mut record = vec![
            result.row_id.to_string(),
            result.text.clone(),
        ];
        
        // Add scores in the same order as labels
        for label in &labels {
            record.push(
                result.scores.get(label)
                    .unwrap_or(&0.0)
                    .to_string()
            );
        }
        
        writer.write_record(&record)?;
    }

    writer.flush()?;
    Ok(())
}