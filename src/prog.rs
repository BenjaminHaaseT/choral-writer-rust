//! Module that contains all of the necessary components for the directed harmonic progression  graph.
use std::collections::{HashMap, HashSet};
use twelve_et::prelude::*;
use twelve_et::SATB;

/// The main struct that implements the directed graph of harmonic progression for a given key.
pub struct HarmonicGraph {
    graph: HashMap<u8, (SATB, HashSet<u8>)>,
}
