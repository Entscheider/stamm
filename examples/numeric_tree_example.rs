extern crate stamm;

extern crate serde_json;

use stamm::numerictree;

use std::io::{stdin, stdout, Write};
use std::io::BufRead;
use std::io::Result;

/// Saves the string `s` to the file `filename`
fn write_str_to_file(s: &str, filename: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;
    let mut f: File = try!(File::create(filename));
    f.write_all(s.as_bytes())
}

fn main() {
    use std::str::FromStr;

    // Generate Trainings-Data
    let data: Vec<(f64, numerictree::NumericTruth)> = (0..100)
        .map(|x|
            // This is the function we want to train on
            if x < 10 || x > 50 {
                (x as f64, numerictree::NumericTruth::NotInClass)
            } else {
                (x as f64, numerictree::NumericTruth::InClass)
        })
        .collect();

    // We want to train a NumericTree with values between 0 and 200, using a maximum depth of 10 and
    // a leaf may have only one item from training
    let tree_param = numerictree::NumericTreeLearnParams::new(0f64, 200f64, 1, 10);
    // Use the parameters to train the tree
    let tree = tree_param.learn_tree(&data[..]);

    // Serialize the tree and save it
    let filename = "learned_tree.js";
    println!("Saved tree to {}", filename);
    let json = serde_json::to_string(&tree).unwrap();
    let _ = write_str_to_file(json.as_str(), filename);
    let tree: numerictree::NumericTree = serde_json::from_str(&json).unwrap();

    // User can now enter some numbers, the tree should predict the class label
    println!(
        "Enter some number, this program predicts the probability of being 10<x<50 using a trained tree"
    );
    let stdin = stdin();
    let mut inp = stdin.lock();
    let mut buffer = String::new();
    print!("> ");
    stdout().flush().unwrap();
    while inp.read_line(&mut buffer).unwrap() > 0 {
        if let Ok(el) = f64::from_str(buffer.trim()) {
            println!(" =>   {}", tree.predict(&el).unwrap().probability);
        } else {
            println!("Invalid input {}", buffer);
        }
        buffer.clear();
        print!("> ");
        stdout().flush().unwrap();
    }
}
