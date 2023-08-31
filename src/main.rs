pub mod prog;

use clap::Parser;
use hound;
use twelve_et::{Pitch, SATB};
use prog::prelude::*;

#[derive(Parser, Debug)]
struct ChoralArgs {
    /// Sets the key for the choral
    #[arg(short, long)]
    key: u8,
    /// Flag to determine whether the key is major or minor, defaults to major
    #[arg(short, long)]
    is_minor: bool,
    /// Determines length of choral in quarter notes
    #[arg(short, long)]
    steps: i32,
    /// The file name to write the choral too
    #[arg(short, long)]
    fname: String,
}

fn main() {
    let choral_args = ChoralArgs::parse();
    let key = choral_args.key;
    let is_minor = choral_args.is_minor;
    let steps = choral_args.steps;

    if !(0..12).contains(&key) {
        eprintln!("invalid key, ensure --key is between 0 and 11 inclusive");
    }
    //
    // let harmonic_progression = prog::harm_prog_graph!(key, is_minor);
    // if let Some(choral) = generate_choral(harmonic_progression, steps) {
    //     let mut choral_satb = vec![];
    //     for (harmony, _) in choral {
    //         let (soprano, soprano_octave, soprano_freq) = (harmony.0.0, harmony.0.1, Pitch::compute_frequency())
    //     }
    // }
}
