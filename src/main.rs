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
    let fname = choral_args.fname;

    if !(0..12).contains(&key) {
        eprintln!("invalid key, ensure --key is between 0 and 11 inclusive");
        std::process::exit(1);
    }

    if let Err(e) = create_choral(key, !is_minor, steps, fname) {
        eprintln!("{e}");
        std::process::exit(1);
    }

}
