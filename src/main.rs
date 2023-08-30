mod prog;

use clap::Parser;
use std::str::FromStr;
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
    println!("{:?}", choral_args);
    println!();
}
